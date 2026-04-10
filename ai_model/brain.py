from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import torch
import chromadb
import hashlib
from datetime import datetime
import re
import json
import os

import networkx as nx
import pickle
import matplotlib.pyplot as plt

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
        
        self.graph_path = "afira_graph.pkl"
        if os.path.exists(self.graph_path):
            with open(self.graph_path, 'rb') as f:
                self.graph = pickle.load(f)
            print("Graph loaded from disk!")
        else:
            self.graph = nx.DiGraph()
        
    def extract_entities_and_relationships(self, user_text):
        prompt = (
            f"<|im_start|>system\n"
            f"You are a knowledge graph extractor. Extract entities and relationships as JSON.\n"
            f"RULES:\n"
            f"1. 'entities' must be a list of names.\n"
            f"2. 'relationships' must be a list of [entity1, relation, entity2].\n"
            f"EXAMPLES:\n"
            f"Text: 'The dog is an animal'\n"
            f"Output: {{\"entities\": [\"dog\", \"animal\"], \"relationships\": [[\"dog\", \"is\", \"animal\"]]}}\n"
            f"Text: 'Our dog name is Pablo'\n"
            f"Output: {{\"entities\": [\"dog\", \"Pablo\"], \"relationships\": [[\"dog\", \"name is\", \"Pablo\"]]}}\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Text: '{user_text}'\n"
            f"Output:<|im_end|>\n"
            f"<|im_start|>assistant\n{{"
        )
        
        response = self.generator(
            prompt,
            max_new_tokens=100,
            do_sample=False,
            temperature=0.1, 
            return_full_text=False
        )
        
        return "{" + response[0]["generated_text"].strip()
    
    def build_graph(self, json_data):
        try:
            json_str = re.sub(r"```json|```", "", json_data).strip()
            
            start_idx = json_str.find("{")
            end_idx = json_str.rfind("}")
            
            if start_idx == -1 or end_idx == -1:
                print(f"Error: No JSON found in model output: {json_data}")
                return False

            json_str = json_str[start_idx:end_idx + 1]
            data = json.loads(json_str)
            
            entities = data.get("entities", [])
            relationships = data.get("relationships", [])
            
            forbidden_terms = {"item1", "item2", "item"}
            if any(term in str(entities).lower() for term in forbidden_terms):
                print("The model generated generic data (item1/2). We ignore it.")
                return False
            clean_entities = [e if isinstance(e, str) else str(e) for e in entities]
            
            self.graph.add_nodes_from(clean_entities)
            for rel in relationships:
                if len(rel) == 3:
                    self.graph.add_edge(rel[0], rel[2], relation=rel[1])
                        
            with open(self.graph_path, 'wb') as f:
                pickle.dump(self.graph, f)
            return True
        except Exception as e:
            print(f"Graph update error: {e}")
            return False
        
    def show_graph(self):
        if self.graph.number_of_nodes() == 0:
            return "Graph is empty. Teach me something first."
        
        plt.figure(figsize=(14, 10))
        plt.clf()
        
        pos = nx.spring_layout(self.graph, k=3.0, iterations=100, seed=42)

        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color="#3498db",
            node_size=2500,
            alpha=0.95
        )
        
        nx.draw_networkx_labels(
            self.graph, pos,
            font_color="white",
            font_size=10,
            font_weight="bold"
        )
        
        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color="#e74c3c",
            arrows=True,
            arrowsize=25,
            arrowstyle="-|>",
            width=2.0,
            connectionstyle="arc3,rad=0.1",
            min_source_margin=30,
            min_target_margin=30
        )
        
        edge_labels = nx.get_edge_attributes(self.graph, "relation")
        nx.draw_networkx_edge_labels(
            self.graph, pos,
            edge_labels=edge_labels,
            font_size=9,
            font_color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
        
        plt.title("Afira's Knowledge Graph", fontsize=14, fontweight="bold", pad=20)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        return "Opening graph window..."
             
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
        
        json_output = self.extract_entities_and_relationships(text)
        print(f"DEBUG -> MODEL OUTPUT: {json_output}")
        if self.build_graph(json_output):
            with open(self.graph_path, 'wb') as f:
                pickle.dump(self.graph, f)
            print("Graph saved to disk!")
        
        self.facts_learned += 1
        return f"I learned something new!\nFact: | {text} |", True
    
    def ask_AI(self, user_question):
        if self.memory.count() == 0 and self.graph.number_of_nodes() == 0:
            return "Memory is empty. I don't know anything yet."

        question_emb = self.embed_model.encode(user_question).tolist()
        results = self.memory.query(query_embeddings=[question_emb], n_results=2)
        
        raw_docs = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        
        filtered_context = [doc for doc, dist in zip(raw_docs, distances) if dist <= 0.55]

        words = re.findall(r'\w+', user_question.lower())
        graph_facts = []
        for node in self.graph.nodes():
            if node.lower() in words:
                for neighbor in self.graph.neighbors(node):
                    rel = self.graph.get_edge_data(node, neighbor).get("relation", "related to")
                    graph_facts.append(f"{node} {rel} {neighbor}")

        if not filtered_context and not graph_facts:
            return "I don't have information about this in my memory. Please teach me first."
        
        combined_context = ". ".join(filtered_context + graph_facts)

        prompt = (
            f"<|im_start|>system\n"
            f"You are a helpful assistant. Answer the question ONLY using the provided facts. "
            f"Be concise. If the facts don't contain the answer, say 'I don't know'.\n"
            f"FACTS: {combined_context}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{user_question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        try:
            sequences = self.generator(
                prompt,
                max_new_tokens=30,
                do_sample=False,
                temperature=0.1, 
                repetition_penalty=1.5,
                return_full_text=False
            )
            answer = sequences[0]["generated_text"].strip()
            return answer.split("<|im_end|>")[0].split("\n")[0].strip()
        except Exception as e:
            return f"I found these facts, but I'm having trouble phrasing it: {combined_context}"