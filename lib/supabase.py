import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

def create_supabase_client():
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Supabase URL and Key must be set in environment variables.")
    supabase: Client = create_client(url, key)
    return supabase
