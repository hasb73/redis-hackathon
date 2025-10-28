from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
import uvicorn
import re

app = FastAPI()
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)  # small, fast: 384-dim

class EmbedRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

def normalize(s: str) -> str:
    s = re.sub(r"\b\d{13,}\b", "<TS>", s)
    s = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<IP>", s)
    s = re.sub(r"0x[0-9a-fA-F]+", "<HEX>", s)
    s = re.sub(r"\d+", "<NUM>", s)
    return s.lower()

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    texts = req.texts
    texts_norm = [normalize(t) for t in texts]
    embeddings = model.encode(texts_norm, show_progress_bar=False, convert_to_numpy=True).tolist()
    return {"embeddings": embeddings}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
