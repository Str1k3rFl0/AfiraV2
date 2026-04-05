from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import chromadb
import hashlib
from datetime import datetime

class AIModel():
    def __init__(self):
        model_name = "utter-project/EuroLLM-1.7B-Instruct"
        
        print("Loading EuroLLM...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=0,
            torch_dtype=torch.float16,
            load_in_8bit=True        
        )
        
        print("Model successfuly loaded!")
        
        self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        self.client = chromadb.PersistentClient(path="./afira_memory")
        self.memory = self.client.get_or_create_collection(name="afira_facts", metadata={"hnsw:space": "cosine"})
        self.facts_learned = self.memory.count()
        
    def teach_AI(self, user_text):
        text = user_text.strip()
        if not text:
            return "Nu mi-ai spus nimic ce sa invat."
        
        fact_id = hashlib.sha1(text.encode()).hexdigest()
        embedding = self.embed_model.encode(text).tolist()
        
        existing_fact = self.memory.get(ids=[fact_id])
        if existing_fact["ids"]:
            return f"Stiam deja /{text}/"
        
        self.memory.add(
            documents=[text],
            embeddings=[embedding],
            ids=[fact_id],
            metadatas=[{"added_at": datetime.now().isoformat()}]
        )
        
        return f"Am invatat ceva nou\nSi anume | {text} |"