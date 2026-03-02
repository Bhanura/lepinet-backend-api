from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import timm
from PIL import Image
import io
from torchvision import transforms
import pandas as pd

app = FastAPI()

# --- 1. CONFIGURATIONS ---
NUM_CLASSES = 245
MODEL_PATH = "butterfly_model_v1.pth" 
CSV_PATH = "sri_lanka_butterflies_245.csv"

# --- 2. LOAD DATA & MODEL (Global Variables) ---
# සර්වර් එක start වෙද්දී මේවා මතකයට (memory) load කරගනී
try:
    df = pd.read_csv(CSV_PATH)
    # Model output index (0-244) එක ID සහ Name වලට map කිරීම
    idx_to_info = {i: {'id': row['butterfly_id'], 'name': row['common_name_english']} for i, row in df.iterrows()}

    # Hugging Face Free Tier සඳහා CPU භාවිතා කිරීම
    device = torch.device("cpu") 
    model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Model or CSV failed to load. Error: {e}")
    model = None

# --- 3. IMAGE PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 4. PREDICT ENDPOINT ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Model එක load වී නැත්නම් error එකක් යැවීම
    if model is None:
        return JSONResponse(status_code=500, content={"error": "AI Server is starting up or missing model files."})

    try:
        # රූපය කියවා ගැනීම (Mobile App එකෙන් එන)
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Tensor බවට හැරවීම
        input_tensor = transform(image).unsqueeze(0)

        # AI Prediction එක ලබා ගැනීම
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        # අගයන් වෙන් කර ගැනීම
        idx = predicted_idx.item()
        conf_score = round(confidence.item(), 4) # දශමස්ථාන 4කට වැටයීම
        
        # ID සහ Name සොයාගැනීම
        species_info = idx_to_info.get(idx, {"id": "unknown", "name": "Unknown"})

        # Success Response එක App එකට යැවීම
        return {
            "species_id": species_info['id'],
            "species_name": species_info['name'],
            "confidence": conf_score
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to process image: {str(e)}"})