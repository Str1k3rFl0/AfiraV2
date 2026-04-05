from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import torch
import chromadb
import hashlib
from datetime import datetime
import re

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
        
        print("Model successfully loaded!")
        
        self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        self.client = chromadb.PersistentClient(path="./afira_memory")
        self.memory = self.client.get_or_create_collection(name="afira_facts", metadata={"hnsw:space": "cosine"})
        self.facts_learned = self.memory.count()
        
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
        
    def teach_AI(self, user_text):
        text = user_text.strip()
        if not text:
            return "You didn't tell me anything to learn.", False
        
        fact_id = hashlib.sha1(text.encode()).hexdigest()
        embedding = self.embed_model.encode(text).tolist()
        
        existing_fact = self.memory.get(ids=[fact_id])
        if existing_fact["ids"]:
            return f"I already knew that: '{text}'", False
        
        self.memory.add(
            documents=[text],
            embeddings=[embedding],
            ids=[fact_id],
            metadatas=[{"added_at": datetime.now().isoformat()}]
        )
        
        self.facts_learned += 1
        return f"I learned something new!\nFact: | {text} |", True
    
    def ask_AI(self, user_question):
        if self.memory.count() == 0:
            return "Memory is empty. Use 'learn: <text>'"

        question_emb = self.embed_model.encode(user_question).tolist()
        results = self.memory.query(query_embeddings=[question_emb], n_results=1)

        if not results["documents"] or not results["documents"][0]:
            return "I don't know that yet."

        context = results["documents"][0][0]

        prompt = (
            f"Context: {context}\n"
            f"Question: {user_question}\n"
            f"Short Answer:"
        )

        try:
            sequences = self.generator(
                prompt,
                max_new_tokens=10,      
                do_sample=False,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            raw_answer = sequences[0]["generated_text"].strip()
            clean_answer = raw_answer.split("\n")[0]
            clean_answer = re.sub(r'[^a-zA-Z0-9\s!\?\.]', '', clean_answer).strip()
            
            if len(clean_answer) < 2 or "I dont know" in clean_answer:
                return f"Based on my memory: {context}"
                
            return clean_answer

        except Exception as e:
            return f"Memory: {context}"