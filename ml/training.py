import os
import io
import tempfile
import requests
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import BaseModel, Field
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

from config import supabase, SUPABASE_URL, REPO_ID, HF_TOKEN
import ml.model as ml_state  # Import global states safely

class TrainParams(BaseModel):
    epochs: int = 5
    learning_rate: float = 0.0001
    batch_size: int = 16
    test_size: float = Field(0.2, ge=0.05, le=0.5)

class OnlineButterflyDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]

def run_fine_tuning_task(params: TrainParams):
    temp_dir = tempfile.mkdtemp()
    model = ml_state.model_instance # Get the active model
    id_to_idx = ml_state.id_to_idx
    
    try:
        print(f"--- Training Started with Config: {params.dict()} ---")
        response = supabase.table('ai_logs').select('id, image_url, final_species_id').eq('training_status', 'ready').execute()
        records = response.data

        if not records:
            print("No new data to train on. Exiting task.")
            return

        image_paths, labels = [], []
        for i, rec in enumerate(records):
            try:
                img_data = requests.get(rec['image_url']).content
                img_path = os.path.join(temp_dir, f"img_{i}.jpg")
                with open(img_path, 'wb') as f: f.write(img_data)
                
                class_idx = id_to_idx.get(rec['final_species_id'])
                if class_idx is not None:
                    image_paths.append(img_path)
                    labels.append(class_idx)
            except Exception as e:
                print(f"Skipping record {rec.get('id', 'N/A')} due to error: {e}")


        if len(image_paths) < 10:
            print(f"Not enough valid images to train. Found {len(image_paths)}, but require at least 10.")
            raise Exception("Not enough valid images to perform a train/val split. Minimum 10 required.")

        # Split data into training and validation sets
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=params.test_size, random_state=42, stratify=labels
        )

        train_dataset = OnlineButterflyDataset(train_paths, train_labels, transform=ml_state.train_transform)
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)

        val_dataset = OnlineButterflyDataset(val_paths, val_labels, transform=ml_state.val_transform)
        eval_loader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False)
        
        # Freezing layers
        for param in model.parameters():
            param.requires_grad = False
        
        if hasattr(model, 'classifier'):
            for param in model.classifier.parameters(): param.requires_grad = True
        elif hasattr(model, 'head'):
            for param in model.head.parameters(): param.requires_grad = True

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.learning_rate)
        criterion = nn.CrossEntropyLoss()
        model.train()

        for epoch in range(params.epochs):
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{params.epochs}, Loss: {running_loss/len(train_loader):.4f}")

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, targets in eval_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

        # --- Metrics & Reporting ---
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"Validation Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # --- Model Versioning and Upload ---
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        version_name = f"v{timestamp}"
        model_path = os.path.join(temp_dir, f"{version_name}.pth")
        torch.save(model.state_dict(), model_path)

        api = HfApi(token=HF_TOKEN)
        repo_model_path = f"models/{version_name}.pth"
        api.upload_file(path_or_fileobj=model_path, path_in_repo=repo_model_path, repo_id=REPO_ID, repo_type="space")
        
        # Upload confusion matrix
        cm_path_in_repo = f"metrics/{version_name}_confusion_matrix.png"
        api.upload_file(path_or_fileobj=buf, path_in_repo=cm_path_in_repo, repo_id=REPO_ID, repo_type="space")
        buf.close()

        # --- Database Logging ---
        model_url = f"https://huggingface.co/spaces/{REPO_ID}/resolve/main/{repo_model_path}"
        metrics_url = f"https://huggingface.co/spaces/{REPO_ID}/resolve/main/{cm_path_in_repo}"

        supabase.table('model_versions').insert({
            "version_name": version_name,
            "file_path": repo_model_path,
            "metrics": {"accuracy": accuracy, "f1_score": f1},
            "config": params.dict(),
            "is_active": False,
            "metrics_url": metrics_url,
            "model_url": model_url
        }).execute()

        # Update status of trained records
        trained_ids = [rec['id'] for rec in records]
        supabase.table('ai_logs').update({'training_status': 'trained'}).in_('id', trained_ids).execute()
        
        print(f"--- Training Complete. Model '{version_name}' saved and logged. ---")

    except Exception as e:
        print(f"An error occurred during the training task: {e}")
    finally:
        # Clean up temp directory
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files: os.remove(os.path.join(root, name))
            for name in dirs: os.rmdir(os.path.join(root, name))
        os.rmdir(temp_dir)
        print("Temporary directory cleaned up.")