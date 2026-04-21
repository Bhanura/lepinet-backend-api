import os
import torch
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd
from config import NUM_CLASSES, CURRENT_MODEL_PATH, MAPPING_PATH, CSV_PATH, supabase
import json

# Global State Variables (වෙනත් ෆයිල් වලට පාවිච්චි කිරීමට)
model_instance = None
idx_to_info = {}
id_to_idx = {}

# Define the same transforms as used during original training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Transformations for fine-tuning
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model_and_data():
    global model_instance, idx_to_info, id_to_idx, transform
    
    # Load species mapping
    with open('species_mapping.json', 'r') as f:
        id_to_idx = json.load(f)
        
    df = pd.read_csv(CSV_PATH)
    idx_to_info = {}
    for _, row in df.iterrows():
        b_id = row['butterfly_id']
        if b_id in id_to_idx:
            idx_to_info[id_to_idx[b_id]] = {'id': b_id, 'name': row['common_name_english']}

    active_model_file = CURRENT_MODEL_PATH
    if supabase:
        res = supabase.table('model_versions').select('file_path').eq('is_active', True).execute()
        if res.data and len(res.data) > 0:
            active_model_file = res.data[0]['file_path']
            
    device = torch.device("cpu") 
    loaded_model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=False, num_classes=NUM_CLASSES)
    loaded_model.load_state_dict(torch.load(active_model_file, map_location=device))
    loaded_model.eval()
    print(f"Model ({active_model_file}) loaded successfully!")
    
    model_instance = loaded_model
    print(f"Model ({os.path.basename(active_model_file) if active_model_file else os.path.basename(CURRENT_MODEL_PATH)}) loaded successfully!")

def load_model(model_path=None):
    """
    Load a model from a file path or use the default model.
    """
    global model_instance, idx_to_info, id_to_idx, transform
    
    # Load species mapping
    with open('species_mapping.json', 'r') as f:
        id_to_idx = json.load(f)
        
    df = pd.read_csv(CSV_PATH)
    idx_to_info = {}
    for _, row in df.iterrows():
        b_id = row['butterfly_id']
        if b_id in id_to_idx:
            idx_to_info[id_to_idx[b_id]] = {'id': b_id, 'name': row['common_name_english']}

    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Model path not found or not provided. Loading default model.")
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
        model.load_state_dict(torch.load(DEFAULT_MODEL_PATH, map_location=device))

    model.to(device)
    model.eval()
    
    # Set global state
    model_instance = model
    print(f"Model ({os.path.basename(model_path) if model_path else os.path.basename(DEFAULT_MODEL_PATH)}) loaded successfully!")

# --- Model Configuration ---
DEFAULT_MODEL_PATH = 'butterfly_model_v1.pth'
CURRENT_MODEL_PATH = 'butterfly_model_v1.pth' # Default model

# Initialize with the active model on startup
from config import supabase

def get_active_model_path():
    try:
        response = supabase.table('model_versions').select('file_path').eq('is_active', True).single().execute()
        if response.data:
            return response.data['file_path']
    except Exception as e:
        print(f"Could not fetch active model, falling back to default. Error: {e}")
    return DEFAULT_MODEL_PATH

load_model(get_active_model_path())