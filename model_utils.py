from transformers import pipeline, AutoTokenizer, AutoModel
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
)

case_labels = ["criminal", "civil", "corporate", "family", "labor", "property", "constitutional"]

def classify_case(text: str):
    result = classifier(text, candidate_labels=case_labels, multi_label=True)
    top_label = result["labels"][0]
    scores = dict(zip(result["labels"], result["scores"]))
    return top_label, scores

ipc_df = pd.read_csv("ipc_sections.csv")
ipc_df["full_text"] = ipc_df["Offense"].fillna('') + ". " + ipc_df["Description"].fillna('')

tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
model = AutoModel.from_pretrained("law-ai/InLegalBERT")

def embed(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def predict_ipc_sections(text: str, top_k: int = 3):
    try:
        case_embedding = embed(text)
        ipc_embeddings = torch.stack([embed(ipc).squeeze() for ipc in ipc_df["full_text"]])
        sims = cosine_similarity(case_embedding, ipc_embeddings)[0]

        top_indices = sims.argsort()[-top_k:][::-1]
        top_sections = ipc_df.iloc[top_indices][["Section", "Offense", "Description"]]

        results = []
        for _, row in top_sections.iterrows():
            results.append(f"Section {row['Section']}: {row['Offense']} — {row['Description']}")

        return "\n\n".join(results)
    except Exception as e:
        return f"Error during IPC prediction: {str(e)}"
