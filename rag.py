import faiss
from transformers import pipeline
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

def getRAG(model_name, tokenizer, instructions):
    emb_model = pipeline("feature-extraction", model=model_name, tokenizer=tokenizer, device=-1)

    embeddings = [torch.tensor(emb_model(inst)[0][0]).mean(0).detach().cpu().numpy() for inst in instructions]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(torch.tensor(embeddings).numpy())
    return index

def getRAG(model_name_RAG, instructions):
    model = SentenceTransformer(model_name_RAG)
    embeddings = model.encode(instructions, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return model, index

def retrieve(query, index, model, instructions, k=3):
    ## Use pipeline
    # q_emb = torch.tensor(emb_model(query)[0][0]).mean(0).detach().cpu().numpy()
    # D, I = index.search(q_emb.reshape(1, -1), k)
    ## USe SentenceTransformer
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    return [instructions[i] for i in I[0]]
