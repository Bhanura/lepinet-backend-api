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

        if not records: return

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
            except Exception: pass

        if len(image_paths) < 10: raise Exception("Not enough valid images to perform a train/val split. Minimum 10 required.")

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
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, targets in eval_loader:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.numpy())
                all_labels.extend(targets.numpy())

        acc = round(accuracy_score(all_labels, all_preds) * 100, 2)
        f1 = round(f1_score(all_labels, all_preds, average='weighted', zero_division=0), 4)

        # Plot
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight')
        img_buf.seek(0)
        plt.close()

        plot_filename = f"eval_plot_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        supabase.storage.from_('eval_plots').upload(file=img_buf.read(), path=plot_filename, file_options={"content-type": "image/png"})
        plot_url = f"{SUPABASE_URL}/storage/v1/object/public/eval_plots/{plot_filename}"

        # Upload & Save
        new_version = f"v1.{datetime.datetime.now().strftime('%m%d%H%M')}"
        new_model_filename = f"butterfly_model_{new_version}.pth"
        
        torch.save(model.state_dict(), new_model_filename)
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=new_model_filename, path_in_repo=new_model_filename,
            repo_id=REPO_ID, repo_type="space", commit_message=f"Added new model version {new_version}"
        )

        # DB Update
        record_ids = [rec['id'] for rec in records]
        supabase.table('ai_logs').update({'training_status': 'trained'}).in_('id', record_ids).execute()
        try: supabase.table('model_versions').update({'is_active': False}).eq('is_active', True).execute()
        except: pass
            
        supabase.table('model_versions').insert({
            'version_name': new_version, 'file_path': new_model_filename,
            'training_image_count': len(image_paths), 'accuracy_score': acc, 'is_active': True
        }).execute()
        supabase.table('model_evaluations').insert({
            'model_version': new_version, 'accuracy': acc, 'f1_score': f1, 'confusion_matrix_url': plot_url
        }).execute()

        print(f"--- Training Complete! Version: {new_version} saved as {new_model_filename} ---")

    except Exception as e:
        print(f"Training Task Failed: {e}")
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)