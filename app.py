from fastapi import FastAPI, File, UploadFile, Header, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from PIL import Image
import io
import os
import tempfile
import requests
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from supabase import create_client, Client
from huggingface_hub import HfApi
import uuid
import datetime

app = FastAPI()

# --- 1. CONFIGURATIONS & SECRETS ---
NUM_CLASSES = 245
MODEL_PATH = "butterfly_model_v1.pth" 
CSV_PATH = "sri_lanka_butterflies_245.csv"
REPO_ID = "bhanura/lepinet-backend" # ඔබේ HF Space නම

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
ADMIN_SECRET = os.environ.get("ADMIN_SECRET")
HF_TOKEN = os.environ.get("HF_TOKEN")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# --- 2. LOAD DATA & MODEL ---
def load_model_and_data():
    global model, idx_to_info, id_to_idx, df
    try:
        df = pd.read_csv(CSV_PATH)
        idx_to_info = {i: {'id': row['butterfly_id'], 'name': row['common_name_english']} for i, row in df.iterrows()}
        id_to_idx = {row['butterfly_id']: i for i, row in df.iterrows()} # Reverse mapping for training

        device = torch.device("cpu") # HF Free tier
        loaded_model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=False, num_classes=NUM_CLASSES)
        loaded_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        loaded_model.eval()
        print("Model loaded successfully!")
        return loaded_model
    except Exception as e:
        print(f"Warning: Model or CSV failed to load. Error: {e}")
        return None

model = load_model_and_data()

# --- 3. PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 4. PREDICT ENDPOINT ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None: return JSONResponse(status_code=500, content={"error": "AI Server missing files."})
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        idx = predicted_idx.item()
        return {
            "species_id": idx_to_info.get(idx, {"id": "unknown"})['id'],
            "species_name": idx_to_info.get(idx, {"name": "Unknown"})['name'],
            "confidence": round(confidence.item(), 4)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- 5. TRAINING HYPERPARAMS MODEL ---
class TrainParams(BaseModel):
    epochs: int = 5
    learning_rate: float = 0.001
    batch_size: int = 16

# --- 6. PYTORCH DATASET ---
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

# --- 7. CONTINUOUS TRAINING LOGIC ---
def run_fine_tuning_task(params: TrainParams):
    global model
    temp_dir = tempfile.mkdtemp()
    
    try:
        print(f"--- Training Started with Config: {params.dict()} ---")
        response = supabase.table('ai_logs').select('id, image_url, final_species_id').eq('training_status', 'ready').execute()
        records = response.data

        if not records:
            print("No records found for training.")
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
                print(f"Failed to download image {rec['image_url']}: {e}")

        if len(image_paths) == 0: raise Exception("No valid images downloaded.")

        dataset = OnlineButterflyDataset(image_paths, labels, transform=train_transform)
        dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
        model.train()

        for epoch in range(params.epochs):
            running_loss = 0.0
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{params.epochs} - Loss: {running_loss/len(dataloader):.4f}")

        model.eval()
        all_preds, all_labels = [], []
        eval_loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False)
        with torch.no_grad():
            for inputs, targets in eval_loader:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.numpy())
                all_labels.extend(targets.numpy())

        acc = round(accuracy_score(all_labels, all_preds) * 100, 2)
        f1 = round(f1_score(all_labels, all_preds, average='weighted', zero_division=0), 4)

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Evaluation Config: Ep:{params.epochs}, LR:{params.learning_rate}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight')
        img_buf.seek(0)
        plt.close()

        plot_filename = f"eval_plot_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        supabase.storage.from_('eval_plots').upload(file=img_buf.read(), path=plot_filename, file_options={"content-type": "image/png"})
        plot_url = f"{SUPABASE_URL}/storage/v1/object/public/eval_plots/{plot_filename}"

        new_version = f"v1.{datetime.datetime.now().strftime('%m%d%H')}"
        torch.save(model.state_dict(), MODEL_PATH)
        
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=MODEL_PATH,
            path_in_repo=MODEL_PATH,
            repo_id=REPO_ID,
            repo_type="space",
            commit_message=f"Auto-update model to {new_version}"
        )

        # -------------------------------------------------------------
        # 7. Database යාවත්කාලීන කිරීම (අලුත්ම සහ නිවැරදි Order එක)
        # -------------------------------------------------------------
        
        # A. ai_logs වල status එක 'trained' කිරීම
        record_ids = [rec['id'] for rec in records]
        supabase.table('ai_logs').update({'training_status': 'trained'}).in_('id', record_ids).execute()
        
        # B. දැනට Active තියෙන පරණ Model Versions ඔක්කොම Inactive කිරීම
        try:
            supabase.table('model_versions').update({'is_active': False}).eq('is_active', True).execute()
        except Exception:
            pass # මුල්ම පාර Train වෙද්දී Active ඒවා නැති නිසා error එකක් ආවොත් මඟහැරීමට
            
        # C. අලුත් Model Version එක model_versions table එකට ඇතුළත් කිරීම (මේක මුලින්ම කළ යුතුයි FK නිසා)
        supabase.table('model_versions').insert({
            'version_name': new_version,
            'file_path': MODEL_PATH,
            'training_image_count': len(image_paths),
            'accuracy_score': acc,
            'is_active': True
        }).execute()

        # D. අලුත් Evaluation එක model_evaluations table එකට ඇතුළත් කිරීම
        supabase.table('model_evaluations').insert({
            'model_version': new_version,
            'accuracy': acc,
            'f1_score': f1,
            'confusion_matrix_url': plot_url
        }).execute()

        print(f"--- Training Complete! Version: {new_version}, Acc: {acc}% ---")

    except Exception as e:
        print(f"Training Task Failed: {e}")
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

# --- 8. TRIGGER ENDPOINT ---
@app.post("/trigger-training")
async def trigger_training(params: TrainParams, background_tasks: BackgroundTasks, authorization: str = Header(None)):
    if authorization != f"Bearer {ADMIN_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    background_tasks.add_task(run_fine_tuning_task, params)
    return {
        "message": "Fine-tuning started successfully.",
        "config": params.dict()
    }