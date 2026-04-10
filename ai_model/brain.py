from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import torch
import chromadb
import os
import pickle
import networkx as nx

from graph_logic import extract_entities_and_relationships, build_graph, show_graph
from features_ai import teach_AI, ask_AI


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

    extract_entities_and_relationships = extract_entities_and_relationships
    build_graph = build_graph
    show_graph = show_graph

    teach_AI = teach_AI
    ask_AI = ask_AI