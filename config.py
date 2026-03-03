import os
from supabase import create_client, Client

NUM_CLASSES = 245
CURRENT_MODEL_PATH = "butterfly_model_v1.pth" 
MAPPING_PATH = "species_mapping.json"
CSV_PATH = "sri_lanka_butterflies_245.csv"
REPO_ID = "bhanura/lepinet-backend" 

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
ADMIN_SECRET = os.environ.get("ADMIN_SECRET")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Global Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None