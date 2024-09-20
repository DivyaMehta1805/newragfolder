# import os
# import tempfile
# from fastapi import FastAPI, File, UploadFile
# from sentence_transformers import SentenceTransformer
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import chromadb
# import PyPDF2
# import uvicorn
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import numpy as np
# import torch
# from sentence_transformers import SentenceTransformer
# persist_directory = "./chroma_db"
# client = chromadb.PersistentClient(path=persist_directory)
# collection = client.get_or_create_collection("documents")
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Download the necessary NLTK data (run this once)
# nltk.download('punkt')
# nltk.download('stopwords')

# app = FastAPI()

# # Initialize Sentence Transformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# import torch
# import torch.nn as nn

# # Create a projection layer
# embedding_dim = model.get_sentence_embedding_dimension()

# # Determine the head size based on the error message
# head_size = 32  # This is the size we see in the error message
# projection = nn.Linear(head_size, embedding_dim)

# def get_multi_head_embeddings(text, num_heads=12):
#     # Tokenize the input text
#     encoded_input = model.tokenize([text])
    
#     # Get the model's transformer
#     transformer = model._first_module().auto_model
    
#     # Forward pass through the model
#     with torch.no_grad():
#         outputs = transformer(**encoded_input, output_hidden_states=True)
    
#     # Get the last layer's hidden states
#     last_hidden_state = outputs.hidden_states[-1]
    
#     # Split the hidden state into multiple heads
#     head_size = last_hidden_state.size(-1) // num_heads
#     multi_head_states = last_hidden_state.view(last_hidden_state.size(0), last_hidden_state.size(1), num_heads, head_size)
    
#     # Average over all tokens to get a single vector per head
#     head_embeddings = multi_head_states.mean(dim=1).squeeze()
    
#     # Project each head embedding to the correct dimensionality
#     projected_embeddings = [projection(head.unsqueeze(0)).squeeze(0).detach().cpu().numpy().tolist() for head in head_embeddings]
    
#     return projected_embeddings

# @app.post("/upload-document")
# async def upload_document(file: UploadFile = File(...)):
# # Extract text from PDF

#     # Split text into chunks
    
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#         content = await file.read()
#         temp_file.write(content)
#         temp_file_path = temp_file.name

#     # Extract text from PDF
#     text = extract_text_from_pdf(temp_file_path)
#     # Generate multi-head embeddings
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
#     embeddings_list = [get_multi_head_embeddings(chunk) for chunk in chunks]

#     # Flatten the list of embeddings and create corresponding documents and metadata
#     flattened_embeddings = []
#     flattened_documents = []
#     flattened_metadatas = []
#     flattened_ids = []

#     for i, (chunk, chunk_embeddings) in enumerate(zip(chunks, embeddings_list)):
#         for j, embedding in enumerate(chunk_embeddings):
#             flattened_embeddings.append(embedding)
#             flattened_documents.append(chunk)
#             flattened_metadatas.append({"source": file.filename, "head": f"head_{j}"})
#             flattened_ids.append(f"doc_{i}_head_{j}")

#     # Ensure all lists have the same length
#     assert len(flattened_embeddings) == len(flattened_documents) == len(flattened_metadatas) == len(flattened_ids)

#     # Clear existing documents in the collection
#     existing_ids = collection.get()["ids"]
#     if existing_ids:
#         collection.delete(ids=existing_ids)

#     # Add new documents to ChromaDB
#     collection.add(
#         documents=flattened_documents,
#         embeddings=flattened_embeddings,
#         metadatas=flattened_metadatas,
#         ids=flattened_ids
#     )

#     # Clean up temporary file
#     os.unlink(temp_file_path)

#     return {"message": f"PDF document processed and {len(flattened_embeddings)} multi-head embeddings stored in ChromaDB"}
# @app.get("/get-documents")
# async def get_documents():
#     results = collection.get()
#     return {
#         "document_count": len(results['ids']),
#         "documents": [
#             {
#                 "id": id,
#                 "content": doc[:100] + "...",  # Show first 100 characters
#                 "metadata": meta,
#                 "embedding_sample": emb[:5] + ["..."]  # Show first 5 values of embedding
#             }
#             for id, doc, meta, emb in zip(results['ids'], results['documents'], results['metadatas'], results['embeddings'])
#         ]
#     }

# def extract_text_from_pdf(file_path):
#     # Extract text from PDF
#     text = ""
#     with open(file_path, 'rb') as file:
#         pdf_reader = PyPDF2.PdfReader(file)
#         for page in pdf_reader.pages:
#             text += page.extract_text() + "\n"
    
#     # Tokenize the text
#     words = word_tokenize(text)
    
#     # Get the set of English stopwords
#     stop_words = set(stopwords.words('english'))
    
#     # Remove stopwords and non-alphabetic tokens
#     filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    
#     # Join the filtered words back into a string
#     filtered_text = ' '.join(filtered_words)
    
#     return filtered_text

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8004)
import os
import json
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import torch.nn as nn

persist_directory = "./chroma_db"
client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_or_create_collection("json")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Download the necessary NLTK data (run this once)
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create a projection layer
embedding_dim = model.get_sentence_embedding_dimension()

# Determine the head size based on the error message
head_size = 32  # This is the size we see in the error message
projection = nn.Linear(head_size, embedding_dim)

def get_multi_head_embeddings(text, num_heads=12):
    # Tokenize the input text
    encoded_input = model.tokenize([text])
    
    # Get the model's transformer
    transformer = model._first_module().auto_model
    
    # Forward pass through the model
    with torch.no_grad():
        outputs = transformer(**encoded_input, output_hidden_states=True)
    
    # Get the last layer's hidden states
    last_hidden_state = outputs.hidden_states[-1]
    
    # Split the hidden state into multiple heads
    head_size = last_hidden_state.size(-1) // num_heads
    multi_head_states = last_hidden_state.view(last_hidden_state.size(0), last_hidden_state.size(1), num_heads, head_size)
    
    # Average over all tokens to get a single vector per head
    head_embeddings = multi_head_states.mean(dim=1).squeeze()
    
    # Project each head embedding to the correct dimensionality
    projected_embeddings = [projection(head.unsqueeze(0)).squeeze(0).detach().cpu().numpy().tolist() for head in head_embeddings]
    
    return projected_embeddings

def process_json_file(file_path):
    # Read JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    flattened_embeddings = []
    flattened_documents = []
    flattened_metadatas = []
    flattened_ids = []

    for i, item in enumerate(data):
        text = item.get('content', '')  # Assuming 'content' is the key for text in your JSON
        chunks = text_splitter.split_text(text)
        
        embeddings_list = [get_multi_head_embeddings(chunk) for chunk in chunks]
        
        for j, (chunk, chunk_embeddings) in enumerate(zip(chunks, embeddings_list)):
            for k, embedding in enumerate(chunk_embeddings):
                flattened_embeddings.append(embedding)
                flattened_documents.append(chunk)
                flattened_metadatas.append({"source": item.get('url', ''), "head": f"head_{k}"})
                flattened_ids.append(f"doc_{i}_chunk_{j}_head_{k}")

    # Ensure all lists have the same length
    assert len(flattened_embeddings) == len(flattened_documents) == len(flattened_metadatas) == len(flattened_ids)

    # Clear existing documents in the collection
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    # Add new documents to ChromaDB
    collection.add(
        documents=flattened_documents,
        embeddings=flattened_embeddings,
        metadatas=flattened_metadatas,
        ids=flattened_ids
    )

    return f"JSON file processed and {len(flattened_embeddings)} multi-head embeddings stored in ChromaDB"

if __name__ == "__main__":
    json_file_path = "url_contents.json"  # Replace with your JSON file path
    result = process_json_file(json_file_path)
    print(result)