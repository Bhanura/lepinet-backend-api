from fastapi import FastAPI, File, UploadFile, Header, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import torch
import timm
from PIL import Image
import io
import os
import requests
from torchvision import transforms
import pandas as pd
from supabase import create_client, Client
from huggingface_hub import HfApi

app = FastAPI()

# --- 1. CONFIGURATIONS & SECRETS ---
NUM_CLASSES = 245
MODEL_PATH = "butterfly_model_v1.pth" 
CSV_PATH = "sri_lanka_butterflies_245.csv"

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
ADMIN_SECRET = os.environ.get("ADMIN_SECRET")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Supabase Client එක සෑදීම
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# --- 2. LOAD DATA & MODEL (Global Variables) ---
def load_model_and_data():
    global model, idx_to_info, df
    try:
        df = pd.read_csv(CSV_PATH)
        idx_to_info = {i: {'id': row['butterfly_id'], 'name': row['common_name_english']} for i, row in df.iterrows()}

        device = torch.device("cpu") 
        loaded_model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=False, num_classes=NUM_CLASSES)
        loaded_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        loaded_model.eval()
        print("Model loaded successfully!")
        return loaded_model
    except Exception as e:
        print(f"Warning: Model or CSV failed to load. Error: {e}")
        return None

model = load_model_and_data()

# --- 3. IMAGE PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 4. PREDICT ENDPOINT (Mobile App එක සඳහා) ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "AI Server is starting up or missing model files."})
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        idx = predicted_idx.item()
        conf_score = round(confidence.item(), 4) 
        species_info = idx_to_info.get(idx, {"id": "unknown", "name": "Unknown"})

        return {
            "species_id": species_info['id'],
            "species_name": species_info['name'],
            "confidence": conf_score
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to process image: {str(e)}"})


# --- 5. CONTINUOUS TRAINING LOGIC (Background Task) ---
def run_fine_tuning_task():
    global model
    try:
        print("--- Training Started ---")
        if not supabase:
            raise Exception("Supabase is not configured.")

        # 1. 'ready' තත්වයේ ඇති AI logs ලබා ගැනීම
        response = supabase.table('ai_logs').select('*').eq('training_status', 'ready').execute()
        records = response.data

        if not records or len(records) == 0:
            print("No new records to train.")
            return

        print(f"Found {len(records)} records for training.")
        
        # 2. මෙතැනදී අපි PyTorch Training Logic එක අනාගතයේදී (Phase 3.5) ලියනවා
        # (Images download කිරීම, DataLoader සෑදීම, Fine-tuning, Evaluation, සහ Model Upload කිරීම)
        
        # දැනට අපි Model එක Train වුණා යැයි සලකා Database එක පමණක් Update කරමු (Test කිරීම සඳහා)
        
        # 3. Database එක Update කිරීම (status = 'trained')
        for record in records:
            supabase.table('ai_logs').update({'training_status': 'trained'}).eq('id', record['id']).execute()
            
        # 4. Model Evaluations එකට බොරු (Dummy) වාර්තාවක් දැමීම (ටෙස්ට් කිරීමට)
        supabase.table('model_evaluations').insert({
            'model_version': 'v1.1_test',
            'accuracy': 95.5,
            'f1_score': 0.94,
            'confusion_matrix_url': 'https://example.com/dummy.png'
        }).execute()

        print("--- Training Completed Successfully ---")

    except Exception as e:
        print(f"Training Task Failed: {e}")

# --- 6. TRIGGER TRAINING ENDPOINT (Web Portal එක සඳහා) ---
@app.post("/trigger-training")
async def trigger_training(background_tasks: BackgroundTasks, authorization: str = Header(None)):
    # 1. Security Check (Admin Password එක හරියටම එවලාද බලනවා)
    if authorization != f"Bearer {ADMIN_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid Admin Secret")

    # 2. Background Task එක ආරම්භ කිරීම
    background_tasks.add_task(run_fine_tuning_task)

    # 3. Admin portal එකට ක්ෂණිකව පිළිතුරක් යැවීම (Timeout නොවීමට)
    return {"message": "Fine-tuning process started in the background.", "status": "processing"}