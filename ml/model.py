import torch
import timm
import json
import pandas as pd
from torchvision import transforms
from config import NUM_CLASSES, CURRENT_MODEL_PATH, MAPPING_PATH, CSV_PATH, supabase

# Global State Variables (වෙනත් ෆයිල් වලට පාවිච්චි කිරීමට)
model_instance = None
idx_to_info = {}
id_to_idx = {}

# Preprocessing Transforms
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

def load_model_and_data():
    global model_instance, idx_to_info, id_to_idx
    try:
        with open(MAPPING_PATH, 'r') as f:
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
    except Exception as e:
        print(f"Warning: Model or Mapping failed to load. Error: {e}")