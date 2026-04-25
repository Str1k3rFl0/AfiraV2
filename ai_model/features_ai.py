import hashlib
import pickle
import re
from datetime import datetime
import json
import os

def teach_AI(self, user_text):
    text = user_text.strip()
    if not text:
        return "You didn't tell me anything to learn.", False
    
    all_sentences = [s.strip() for s in re.split(r'[.!?\n]+', text) if s.strip()]
    
    learned_count = 0
    already_known_count = 0
    
    batch_size = 5
    
    for i in range(0, len(all_sentences), batch_size):
        batch = all_sentences[i : i + batch_size]
        
        new_sentences = []
        for s in batch:
            fact_id = hashlib.sha1(s.encode()).hexdigest()
            if not self.memory.get(ids=[fact_id])["ids"]:
                new_sentences.append(s)
            else:
                already_known_count += 1
        
        if not new_sentences:
            continue

        embeddings = self.embed_model.encode(new_sentences).tolist()
        ids = [hashlib.sha1(s.encode()).hexdigest() for s in new_sentences]
        self.memory.add(
            documents=new_sentences,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{"timestamp": str(datetime.now())} for _ in new_sentences]
        )

        print(f"Processing batch {i//batch_size + 1} ({len(new_sentences)} facts)...")
        json_output = self.extract_entities_and_relationships(new_sentences)
        self.build_graph(json_output)
        
        learned_count += len(new_sentences)

    if learned_count > 0:
        with open(self.graph_path, 'wb') as f:
            pickle.dump(self.graph, f)
        self.facts_learned += learned_count
        return f"Finished! Learned {learned_count} new facts in record time.", True
    
    print(f"Stats: {learned_count} learned, {already_known_count} skipped (already known).")
    
    return "I already knew all of that!", False
    
def learn_document(self, file_path):
    filename = ""
    file_path = "./documents/extracted_docs/" + file_path
    
    if not os.path.exists(file_path):
        return f"File not found. | {file_path} |", False
    
    with open(file_path, "r", encoding="utf-8") as f:
        document_text = f.read()
    
    if not document_text.strip():
        return "The document is empty.", False
    
    response, success = self.teach_AI(document_text)
    if success:
        return f"Document processed successfully!\n{response}", True
    else:
        return response, False


def ask_AI(self, user_question):
    if self.memory.count() == 0 and self.graph.number_of_nodes() == 0:
        return "Memory is empty. I don't know anything yet."

    question_emb = self.embed_model.encode(user_question).tolist()
    results = self.memory.query(query_embeddings=[question_emb], n_results=2)
    
    raw_docs = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []
    
    filtered_context = [doc for doc, dist in zip(raw_docs, distances) if dist <= 0.65]
    stopwords = {"what", "is", "a", "an", "the", "who", "where", "why", "how", "are", "do", "does", "did", "to", "in", "on", "of", "for", "and"}
    
    graph_facts = []
    user_q_lower = user_question.lower()
    
    for node in self.graph.nodes():
        node_lower = str(node).lower()
        if len(node_lower) > 2 and node_lower not in stopwords and node_lower in user_q_lower:
            for neighbor in self.graph.neighbors(node):
                rel = self.graph.get_edge_data(node, neighbor).get("relation", "related to")
                graph_facts.append(f"{node} {rel} {neighbor}")

    if not filtered_context and not graph_facts:
        return "I don't have information about this in my memory. Please teach me first."
    
    combined_context = ". ".join(filtered_context + graph_facts)

    few_shot_example = (
        f"<|im_start|>user\n"
        f"FACTS: The sky is pink. Grass is blue.\n"
        f"QUESTION: What color is the sky?<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"The sky is pink.<|im_end|>\n"
    )

    prompt = (
        f"<|im_start|>system\n"
        f"You are Afira, a strict retrieval assistant. Answer ONLY based on the FACTS provided. "
        f"If the answer to the QUESTION is not explicitly in the FACTS, you MUST reply exactly with: 'I don't know based on my memory.' "
        f"Do not use your internal knowledge.\n<|im_end|>\n"
        f"{few_shot_example}"
        f"<|im_start|>user\n"
        f"FACTS: {combined_context}\n"
        f"QUESTION: {user_question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    try:
        print(f"DEBUG -> CONTEXT: {combined_context}")
        
        sequences = self.generator(
            prompt,
            max_new_tokens=100,
            do_sample=False,
            repetition_penalty=1.0,
            return_full_text=False
        )
        answer = sequences[0]["generated_text"].strip()
        return answer.split("<|im_end|>")[0].split("\n")[0].strip()
    except Exception as e:
        return f"I found these facts, but I'm having trouble phrasing it: {combined_context}"
    
def forget_facts(self, user_text):
    text = user_text.strip()
    if not text:
        return "You didn't tell me anything to forget."
    
    fact_id = hashlib.sha1(user_text.encode()).hexdigest()
    existing_fact = self.memory.get(ids=[fact_id])
    if not existing_fact["ids"]:
        return "Nothing to forget."
    
    self.memory.delete(ids=[fact_id])
    self.facts_learned = max(0, self.facts_learned - 1)
    
    try:
        json_output = self.extract_entities_and_relationships(text)
        data = json.loads(json_output)
        
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])
        
        for rel in relationships:
            if len(rel) >= 3:
                u, v = rel[0], rel[2]
                if self.graph.has_edge(u, v):
                    self.graph.remove_edge(u, v)
                    
        for entity in entities:
            if self.graph.has_node(entity) and self.graph.degree(entity) == 0:
                self.graph.remove_node(entity)
                
        with open(self.graph_path, 'wb') as f:
            pickle.dump(self.graph, f)
            
    except Exception as e:
        print(f"Graph cleanup error: {e}")
        
    return f"I have forgotten | {text} | and updated my knowledge graph.", True

def edit_facts(self, old_text, new_text):
    old_text = old_text.strip()
    new_text = new_text.strip()
    if not old_text or not new_text:
        return "You didn't tell me anything to edit."
    
    old_id = hashlib.sha1(old_text.encode()).hexdigest()
    existing_fact = self.memory.get(ids=[old_id])
    
    if not existing_fact["ids"]:
        return f"I couldn't find the fact: | {old_text} |", False
    
    self.forget_facts(old_text)
    response, success = self.teach_AI(new_text)
    
    if success:
        return f"Done! I've updated my memory.\nFrom: {old_text}\nTo: {new_text}", True
    else:
        return "Something went wrong while learning the new fact.", False

    
def show_all_facts(self, search_val):
    if self.memory.count() == 0:
        return "Nothing to show here!"
    
    facts = self.memory.get()
    if not facts["documents"] or not facts["documents"]:
        return "Nothing to show here!"
    
    try:
        threshold = int(search_val)
        is_numeric = True
    except ValueError:
        is_numeric = False
        
    docs = facts["documents"]
    
    if is_numeric:
        limit = len(docs) if threshold == 0 else threshold
        docs_to_show = docs[:limit]
        header = f"Showing first {len(docs_to_show)} facts:\n\n"
    else:
        query = str(search_val).lower()
        docs_to_show = [d for d in docs if query in d.lower()]
        header = f"Search results for | {search_val} |:\n\n"
        if not docs_to_show:
            return f"I couldn't find any facts containing '{search_val}'."
        
    final_text = header
    for i, doc in enumerate(docs_to_show, 1):
        final_text += f"{i}. {doc}\n"
        
        
    return final_text