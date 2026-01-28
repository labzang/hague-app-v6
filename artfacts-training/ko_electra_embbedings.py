from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List

class KoElectraEmbeddings(Embeddings):
    def __init__(self, model_path: str = "./koelectra_orchestrator_finetuned"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # [CLS] 토큰 사용 (가장 일반적)
                emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            embeddings.append(emb.tolist())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
