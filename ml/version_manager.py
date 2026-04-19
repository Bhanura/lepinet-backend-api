import os
from huggingface_hub import HfApi, hf_hub_download
import torch

from config import supabase, HF_TOKEN, REPO_ID
import ml.model as ml_state # Active model state එක import කරගැනීම

def switch_active_model(version_name: str, file_path: str):
    """
    Switch the currently active model in the application.
    1. Download the target model from Hugging Face.
    2. Load it into memory.
    3. Update the database to set the new active version.
    """
    print(f"--- Switching to model version: {version_name} ---")
    
    # 1. Hugging Face වෙතින් මොඩලය බාගත කිරීම
    try:
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=file_path,
            repo_type="space",
            token=HF_TOKEN
        )
    except Exception as e:
        raise Exception(f"Failed to download model from Hugging Face: {e}")

    # 2. මොඩලය මතකය (memory) වෙත ஏற்றிக்கொள்ளுதல் (load)
    # ml_state.model_instance හි ව්‍යුහය (architecture) නොවෙනස්ව පවතින බව උපකල්පනය කෙරේ
    ml_state.model_instance.load_state_dict(torch.load(model_path))
    ml_state.model_instance.eval() # Set to evaluation mode
    print(f"--- Model {version_name} loaded into memory successfully ---")

    # 3. දත්ත සමුදාය යාවත්කාලීන කිරීම
    # පළමුව, දැනට active ඇති සියලුම මොඩල inactive කිරීම
    supabase.table('model_versions').update({'is_active': False}).eq('is_active', True).execute()
    
    # ඉන්පසු, ඉලක්කගත මොඩලය active කිරීම
    response = supabase.table('model_versions').update({'is_active': True}).eq('version_name', version_name).execute()
    
    if not response.data:
        raise Exception("Failed to update model status in the database.")
        
    print(f"--- Database updated. {version_name} is now the active model. ---")
    return {"message": f"Successfully switched to model version {version_name}."}


def delete_model_version(version_name: str, file_path: str):
    """
    Permanently delete a model version.
    1. Delete the model file from the Hugging Face repository.
    2. Delete the evaluation plot from Supabase Storage.
    3. Delete the corresponding records from the database.
    """
    print(f"--- Deleting model version: {version_name} ---")
    api = HfApi(token=HF_TOKEN)

    # 1. Hugging Face වෙතින් ගොනුව මකා දැමීම
    try:
        api.delete_file(
            path_in_repo=file_path,
            repo_id=REPO_ID,
            repo_type="space",
            commit_message=f"Delete model version {version_name}"
        )
        print(f"--- Deleted {file_path} from Hugging Face repository. ---")
    except Exception as e:
        print(f"Could not delete file from Hugging Face (it may already be gone): {e}")

    # 2. Supabase Storage වෙතින් Confusion Matrix රූපය මකා දැමීම
    try:
        eval_response = supabase.table('model_evaluations').select('confusion_matrix_url').eq('model_version', version_name).single().execute()
        
        if eval_response.data and eval_response.data.get('confusion_matrix_url'):
            url = eval_response.data['confusion_matrix_url']
            plot_filename = url.split('/')[-1]
            
            if plot_filename:
                supabase.storage.from_('eval_plots').remove([plot_filename])
                print(f"--- Deleted plot {plot_filename} from Supabase Storage. ---")

    except Exception as e:
        print(f"Could not delete plot from Supabase Storage (it may not exist): {e}")

    # 3. දත්ත සමුදායෙන් අදාළ වාර්තා මකා දැමීම
    supabase.table('model_evaluations').delete().eq('model_version', version_name).execute()
    response = supabase.table('model_versions').delete().eq('version_name', version_name).execute()
    
    if not response.data:
        raise Exception("Model version not found in the database.")

    print(f"--- Deleted records for {version_name} from the database. ---")
    return {"message": f"Successfully deleted model version {version_name}."}
