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
import json
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from supabase import create_client, Client
from huggingface_hub import HfApi
import datetime

app = FastAPI()

# --- 1. CONFIGURATIONS & SECRETS ---
NUM_CLASSES = 245
CURRENT_MODEL_PATH = "butterfly_model_v1.pth" # දැනට Active එක (ආරම්භයේදී)
MAPPING_PATH = "species_mapping.json"
CSV_PATH = "sri_lanka_butterflies_245.csv"
REPO_ID = "bhanura/lepinet-backend" 

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
ADMIN_SECRET = os.environ.get("ADMIN_SECRET")
HF_TOKEN = os.environ.get("HF_TOKEN")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# --- 2. LOAD DATA & MODEL ---
def load_model_and_data():
    global model, idx_to_info, id_to_idx
    try:
        # JSON Mapping එක කියවීම (නව ක්‍රමය)
        with open(MAPPING_PATH, 'r') as f:
            id_to_idx = json.load(f)
            
        # CSV එකෙන් නම් ටික පමණක් ගැනීම (UI එකට යවන්න)
        df = pd.read_csv(CSV_PATH)
        idx_to_info = {}
        for _, row in df.iterrows():
            b_id = row['butterfly_id']
            if b_id in id_to_idx:
                idx_to_info[id_to_idx[b_id]] = {'id': b_id, 'name': row['common_name_english']}

        # දැනට Active Model එක Database එකෙන් සොයා ගැනීම
        active_model_file = CURRENT_MODEL_PATH
        if supabase:
            res = supabase.table('model_versions').select('file_path').eq('is_active', True).execute()
            if res.data and len(res.data) > 0:
                active_model_file = res.data[0]['file_path']
                
        # Model එක Load කිරීම
        device = torch.device("cpu") 
        loaded_model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=False, num_classes=NUM_CLASSES)
        loaded_model.load_state_dict(torch.load(active_model_file, map_location=device))
        loaded_model.eval()
        print(f"Model ({active_model_file}) loaded successfully!")
        return loaded_model
    except Exception as e:
        print(f"Warning: Model or Mapping failed to load. Error: {e}")
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
    learning_rate: float = 0.0001 # Catastrophic Forgetting වලක්වන්න අගය අඩු කළා
    batch_size: int = 16

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

# --- 6. CONTINUOUS TRAINING LOGIC (Catastrophic Forgetting Prevention) ---
def run_fine_tuning_task(params: TrainParams):
    global model
    temp_dir = tempfile.mkdtemp()
    
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
                
                class_idx = id_to_idx.get(rec['final_species_id']) # JSON එකෙන් අංකය ගැනීම
                if class_idx is not None:
                    image_paths.append(img_path)
                    labels.append(class_idx)
            except Exception as e:
                pass

        if len(image_paths) == 0: raise Exception("No valid images downloaded.")

        dataset = OnlineButterflyDataset(image_paths, labels, transform=train_transform)
        dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True)
        
        # --- CATASTROPHIC FORGETTING PREVENTION (Layer Freezing) ---
        # මුල් ස්තර (Features) වෙනස් වීම නවත්වා අවසන් Classifier එක පමණක් ඉගෙනගැනීමට ඉඩදීම
        for param in model.parameters():
            param.requires_grad = False
        
        # MobileNetV4 හි අවසන් Classifier layer එක පමණක් Train වීමට ඇරීම
        if hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = True

        # Train වෙන්න දාපු layers ටික විතරක් Optimizer එකට දීම
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.learning_rate)
        criterion = nn.CrossEntropyLoss()
        model.train()

        for epoch in range(params.epochs):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

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

        # CM Plot
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

        # --- EXPLICIT VERSIONING: අලුත් නමකින් Model එක Save කිරීම ---
        new_version = f"v1.{datetime.datetime.now().strftime('%m%d%H%M')}"
        new_model_filename = f"butterfly_model_{new_version}.pth" # වෙනම ෆයිල් එකක් හැදේ
        
        torch.save(model.state_dict(), new_model_filename)
        
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=new_model_filename,
            path_in_repo=new_model_filename,
            repo_id=REPO_ID,
            repo_type="space",
            commit_message=f"Added new model version {new_version}"
        )

        # DB Updates
        record_ids = [rec['id'] for rec in records]
        supabase.table('ai_logs').update({'training_status': 'trained'}).in_('id', record_ids).execute()
        
        try:
            supabase.table('model_versions').update({'is_active': False}).eq('is_active', True).execute()
        except: pass
            
        supabase.table('model_versions').insert({
            'version_name': new_version,
            'file_path': new_model_filename, # අලුත් ෆයිල් නම DB එකට යයි
            'training_image_count': len(image_paths),
            'accuracy_score': acc,
            'is_active': True
        }).execute()

        supabase.table('model_evaluations').insert({
            'model_version': new_version,
            'accuracy': acc,
            'f1_score': f1,
            'confusion_matrix_url': plot_url
        }).execute()

        print(f"--- Training Complete! Version: {new_version} saved as {new_model_filename} ---")

    except Exception as e:
        print(f"Training Task Failed: {e}")
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/trigger-training")
async def trigger_training(params: TrainParams, background_tasks: BackgroundTasks, authorization: str = Header(None)):
    if authorization != f"Bearer {ADMIN_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    background_tasks.add_task(run_fine_tuning_task, params)
    return {"message": "Fine-tuning started", "config": params.dict()}