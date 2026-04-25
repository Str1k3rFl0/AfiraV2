import re
import json
import pickle
import networkx as nx
import matplotlib.pyplot as plt


def extract_entities_and_relationships(self, user_text):
    prompt = (
        f"<|im_start|>system\n"
        f"You are a knowledge graph extractor. Extract entities and relationships as JSON.\n"
        f"CRITICAL RULES:\n"
        f"1. Every element used in 'relationships' MUST be present in the 'entities' list.\n"
        f"2. Be specific. If 'A' is the creator of 'B', entities are ['A', 'B'] and relationship is [['A', 'creator of', 'B']].\n"
        f"3. Return ONLY valid JSON.\n"
        f"EXAMPLES:\n"
        f"Text: 'The creator of Afira is Flavius.'\n"
        f"Output: {{\"entities\": [\"Afira\", \"Flavius\"], \"relationships\": [[\"Flavius\", \"is creator of\", \"Afira\"]]}}\n"
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
        max_new_tokens=150,
        do_sample=False,
        temperature=0.1, 
        return_full_text=False
    )
    
    generated = response[0]["generated_text"].strip()
    if not generated.endswith("}"):
        generated += "}"
        
    full_json = "{" + generated
    return full_json


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
                    
        # with open(self.graph_path, 'wb') as f:
        #     pickle.dump(self.graph, f)
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