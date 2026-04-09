# Afira AI: Minimalist RAG Implementation

Afira is lightweight AI assistant build to demonstrate how small language models can be "taught" specific facts in real-time using Retrieval-Augmented Generation (RAG).

Unlike standard AI models that rely only on their pre-training, Afira uses a local vector database to store and retrieve specific information provided by the user.

## Features
  * Brain: Powered by EuroLLM-1.7B-Instruct, a compact but capable model.
  * Memory: Uses ChromaDB for persistent local storage of facts.
  * Embeddings: Utilizes sentence-transformers/all-MiniLM-L6-v2 for high-speed semantic search.
  * GUI: A clean, dark-themed desktop interface built with Tkinter.
  * 8-bit Quantization: Optimized to run on consumer-grade GPUs with low VRAM.

## How it works!
Afira operates on a "Learn & Ask" workflow:
  1. Learn: When you type learn: <fact>, the text is converted into a mathematical vector (embedding) and stored in ChromaDB.
  2. Ask: When you ask a question using ?: <question>, the system searches the database for the most relevant fact.
  3. Restricted Generation: The AI is strictly instructed to answer only based on the retrieved context, preventing common hallucinations.

## Technical Stack
  * Language: Python
  * LLM Framework: Transformers (Hugging Face)
  * Database: ChromaDB
  * GUI: Tkinter
  * Model: EuroLLM-1.7B-Instruct (Quantized in 8-bit)

## Installation && Setup
  1. Clone the repo:
       git clone https://github.com/your-username/afira-ai.git
       cd afira-ai

  2. Install dependecies:
       pip install torch transformers accelerate bitsandbytes sentence-transformers chromadb

  3. Run the application:
       python app/app.py

## Exemple Usage
  * User: learn: Our dog's name is Bobita.
  * AI: I learned something new! Fact: | Our dog's name is Bobita. |
  * User: ?: What is our dog's name?
  * AI: Bobita
