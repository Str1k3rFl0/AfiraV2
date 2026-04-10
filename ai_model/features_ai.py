import hashlib
import pickle
import re
from datetime import datetime


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