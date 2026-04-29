import fitz
import re

extrated_doc_filename = "medical_facts_training_biology_6.txt"
doc_path = "./documents/docs/biology_quantum_mechanisms_350.pdf"
destination = f"./documents/extracted_docs/{extrated_doc_filename}"

def rebuild_paragraphs(text):
    lines = text.split("\n")
    current = ""
    paragraphs = []
    
    for line in lines:
        line = line.strip()
        if not line:
            if current:
                paragraphs.append(current.strip())
                current = ""
            continue
        
        if current.endswith("-"):
            current = current[:-1] + line
        else:
            current += " " + line
            
    if current:
        paragraphs.append(current.strip())
        
    return " ".join(paragraphs)

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


doc = fitz.open(doc_path)

with open(destination, "w", encoding="utf-8") as f:
    for page in doc:
        raw_text = page.get_text()

        text = rebuild_paragraphs(raw_text)
        sentences = split_sentences(text)

        for s in sentences:
            f.write(s + "\n")

doc.close()