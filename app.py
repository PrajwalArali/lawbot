from fastapi import FastAPI
from pydantic import BaseModel
from model_utils import classify_case, predict_ipc_sections

app = FastAPI()

class CaseRequest(BaseModel):
    description: str

@app.get("/")
def home():
    return {"message": "LawBot API is live"}

@app.post("/analyze/")
def analyze_case(case: CaseRequest):
    label, scores = classify_case(case.description)
    ipc = predict_ipc_sections(case.description)
    return {
        "predicted_case_type": label,
        "case_type_scores": scores,
        "ipc_sections": ipc
    }
