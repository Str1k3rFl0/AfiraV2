import os

documents = "./documents/extracted_docs/"
new_namefile = "./documents/extracted_docs/all_medical_facts_training.txt"

with open(new_namefile, "w", encoding="utf-8") as f:
    for filename in os.listdir(documents):
        if filename.endswith(".txt"):
            with open(os.path.join(documents, filename), "r", encoding="utf-8") as infile:
                f.write(infile.read())                