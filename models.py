from transformers import AutoTokenizer, AutoModel, RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch
from db import store_embeddings, retrieve_embeddings

# Initialize the tokenizer and model for document ingestion
ingest_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
ingest_model = AutoModel.from_pretrained("bert-base-uncased")

# Initialize the tokenizer, retriever, and model for Q&A
qa_tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
qa_retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)
qa_model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=qa_retriever)

def ingest_document(document):
    # Tokenize the document
    inputs = ingest_tokenizer(document, return_tensors="pt")
    # Generate embeddings
    outputs = ingest_model(**inputs)
    embeddings = outputs.last_hidden_state
    # Store embeddings in the database
    store_embeddings(embeddings)

def answer_question(question, documents):
    # Encode the question
    inputs = qa_tokenizer(question, return_tensors="pt")
    # Retrieve relevant embeddings
    retrieved_embeddings = retrieve_embeddings(documents)
    # Generate answer
    generated = qa_model.generate(inputs["input_ids"], retrieved_embeddings)
    answer = qa_tokenizer.decode(generated[0], skip_special_tokens=True)
    return answer