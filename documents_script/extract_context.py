import fitz

extrated_doc_filename = "beginner_medicine_facts2.txt"
doc_path = "./documents/docs/beginner_medicine_facts2.pdf"
destination = f"./documents/extracted_docs/{extrated_doc_filename}"

doc = fitz.open(doc_path)
first_page = doc[0]
text = first_page.get_text()

document_title = text.split("\n")[0].strip().upper()
#print(document_title)
            
with open(destination, "a", encoding="utf-8") as f:
    f.write(document_title + "\n")
    for line in text.split("\n")[1:]:
        if line.strip():
            f.write(line + "\n")