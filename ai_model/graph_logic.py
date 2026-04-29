import re
import json
import pickle
import networkx as nx
import matplotlib.pyplot as plt


def _regex_extract(sentence):
    s = sentence.strip().rstrip('.!?')
    s = re.sub(r'^(The|A|An)\s+', '', s, flags=re.IGNORECASE)

    m = re.match(
        r'^(.+?)\s+(is|are|was|were|has|have|contains|includes)\s+(?:a |an |the )?(.+)$',
        s, re.IGNORECASE
    )
    if m:
        subj = m.group(1).strip().lower()
        verb = m.group(2).strip().lower()
        obj  = re.sub(r'^(a |an |the )', '', m.group(3).strip().lower())
        if len(subj) > 1 and len(obj) > 1:
            return [subj, obj], [[subj, verb, obj]]

    m = re.match(r'^(.+?)\s+([a-z]+(?:s|ed|es)?)\s+(?:a |an |the )?(.+)$', s, re.IGNORECASE)
    if m:
        subj = m.group(1).strip().lower()
        verb = m.group(2).strip().lower()
        obj  = re.sub(r'^(a |an |the )', '', m.group(3).strip().lower())
        stopverbs = {'please','very','also','just','only','even','still'}
        if len(subj) > 1 and len(obj) > 1 and verb not in stopverbs:
            return [subj, obj], [[subj, verb, obj]]

    words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', s)]
    stopwords = {'the','and','for','are','but','not','you','all','can','was',
                 'one','our','out','get','has','how','its','may','new','now'}
    return [w for w in words if w not in stopwords][:4], []


def _parse_llm_json(generated):
    depth, start = 0, None
    for idx, ch in enumerate(generated):
        if ch == '{':
            if depth == 0:
                start = idx
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                candidate = generated[start:idx+1]
                try:
                    parsed = json.loads(candidate)
                    if "entities" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    pass
                start = None
    return None


def extract_entities_and_relationships(self, sentences):
    text_block = "\n".join([f"- {s}" for s in sentences])

    prompt = (
        f"<|im_start|>system\n"
        f"You are a JSON knowledge graph extractor. Output ONLY valid JSON, nothing else.\n"
        f"Format: {{\"entities\": [\"A\", \"B\"], \"relationships\": [[\"A\", \"verb\", \"B\"]]}}\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Sentences:\n- The heart pumps blood.\n- Lungs help breathing.\n"
        f"JSON:<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{{\"entities\": [\"heart\", \"blood\", \"lungs\", \"breathing\"], \"relationships\": [[\"heart\", \"pumps\", \"blood\"], [\"lungs\", \"help\", \"breathing\"]]}}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Sentences:\n{text_block}\n"
        f"JSON:<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{{"  
    )

    try:
        response = self.generator(
            prompt,
            max_new_tokens=1024,
            do_sample=False,
            repetition_penalty=1.1, 
            return_full_text=False
        )
        
        generated = "{" + response[0]["generated_text"].strip()
        print(f"[Graph] LLM output: {generated[:450]}")

        parsed = _parse_llm_json(generated)
        if parsed:
            print(f"[Graph] LLM success: {parsed}")
            return json.dumps(parsed)
        else:
            print("[Graph] LLM failed, using regex fallback.")

    except Exception as e:
        print(f"[Graph] Generator error: {e}")

    all_entities, all_relationships = [], []
    for sentence in sentences:
        ents, rels = _regex_extract(sentence)
        all_entities.extend(ents)
        all_relationships.extend(rels)
    all_entities = list(dict.fromkeys(all_entities))
    result = {"entities": all_entities, "relationships": all_relationships}
    print(f"[Graph] Regex fallback result: {result}")
    return json.dumps(result)


def build_graph(self, json_data):
    try:
        depth, start, data = 0, None, None
        for idx, ch in enumerate(json_data):
            if ch == '{':
                if depth == 0:
                    start = idx
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = json_data[start:idx+1]
                    try:
                        data = json.loads(candidate)
                        break
                    except json.JSONDecodeError:
                        pass
                    start = None

        if data is None:
            print(f"[Graph] build_graph: no valid JSON in: {json_data[:80]}")
            return False

        entities = data.get("entities", [])
        relationships = data.get("relationships", [])

        if not entities and not relationships:
            return False

        clean_entities = [e if isinstance(e, str) else str(e) for e in entities]
        self.graph.add_nodes_from(clean_entities)
        for rel in relationships:
            if len(rel) == 3:
                self.graph.add_edge(str(rel[0]), str(rel[2]), relation=str(rel[1]))

        #print(f"[Graph] Nodes: {list(self.graph.nodes())}")

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

    nx.draw_networkx_nodes(self.graph, pos, node_color="#3498db", node_size=2500, alpha=0.95)
    nx.draw_networkx_labels(self.graph, pos, font_color="white", font_size=10, font_weight="bold")
    nx.draw_networkx_edges(
        self.graph, pos, edge_color="#e74c3c", arrows=True,
        arrowsize=25, arrowstyle="-|>", width=2.0,
        connectionstyle="arc3,rad=0.1",
        min_source_margin=30, min_target_margin=30
    )
    edge_labels = nx.get_edge_attributes(self.graph, "relation")
    nx.draw_networkx_edge_labels(
        self.graph, pos, edge_labels=edge_labels,
        font_size=9, font_color="#2c3e50",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    plt.title("Afira's Knowledge Graph", fontsize=14, fontweight="bold", pad=20)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return "Opening graph window..."