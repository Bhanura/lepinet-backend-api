from fastapi import FastAPI
from api.routes import router
from ml.model import load_model_and_data

# FastAPI සර්වර් එක ආරම්භ කිරීම
app = FastAPI(title="LepiNet AI Backend", version="2.0")

# සර්වර් එක on වෙද්දී මොඩලය මතකයට Load කිරීම
@app.on_event("startup")
def startup_event():
    load_model_and_data()

# Routes ටික ඇතුළත් කිරීම
app.include_router(router)