import io
import torch
from fastapi import APIRouter, File, UploadFile, Header, HTTPException, BackgroundTasks, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

from config import ADMIN_SECRET
import ml.model as ml_state
from ml.training import run_fine_tuning_task, TrainParams
from ml.version_manager import switch_active_model, delete_model_version
from lib.supabase import create_supabase_client

supabase = create_supabase_client()

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if ml_state.model_instance is None: 
        return JSONResponse(status_code=500, content={"error": "AI Server missing files."})
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = ml_state.transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = ml_state.model_instance(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        idx = predicted_idx.item()
        return {
            "species_id": ml_state.idx_to_info.get(idx, {"id": "unknown"})['id'],
            "species_name": ml_state.idx_to_info.get(idx, {"name": "Unknown"})['name'],
            "confidence": round(confidence.item(), 4)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/trigger-training")
async def trigger_training(params: TrainParams, background_tasks: BackgroundTasks, authorization: str = Header(None)):
    if authorization != f"Bearer {ADMIN_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    background_tasks.add_task(run_fine_tuning_task, params)
    return {"message": "Fine-tuning started successfully.", "config": params.dict()}

@router.post("/set-active-model")
async def set_active_model(
    authorization: str = Header(None),
    version_name: str = Body(..., embed=True),
    file_path: str = Body(..., embed=True)
):
    if authorization != f"Bearer {ADMIN_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        result = switch_active_model(version_name, file_path)
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.delete("/delete-model")
async def delete_model(
    authorization: str = Header(None),
    version_name: str = Body(..., embed=True),
    file_path: str = Body(..., embed=True)
):
    if authorization != f"Bearer {ADMIN_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    active_model_check = supabase.table('model_versions').select('is_active').eq('version_name', version_name).single().execute()
    if active_model_check.data and active_model_check.data['is_active']:
        raise HTTPException(status_code=400, detail="Cannot delete the currently active model. Please switch to another model first.")

    try:
        result = delete_model_version(version_name, file_path)
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})