import fitz

extrated_doc_filename = "medical_facts_training_9.txt"
doc_path = "./documents/docs/reproductive_health_training.pdf"
destination = f"./documents/extracted_docs/{extrated_doc_filename}"

doc = fitz.open(doc_path)
i = 0

#document_title = text.split("\n")[0].strip().upper()
#print(document_title)
            
# with open(destination, "a", encoding="utf-8") as f:
#     for line in text.split("\n"):
#         if line.strip():
#             f.write(line + "\n")

while i < len(doc):
    page = doc[i]
    text = page.get_text()
    with open(destination, "a", encoding="utf-8") as f:
        for line in text.split("\n"):
            if line.strip():
                f.write(line + "\n")
    i += 1
    
doc.close()