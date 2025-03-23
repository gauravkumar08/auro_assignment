# Document Management and RAG-based Q&A Application

## Overview
This project is a FastAPI-based backend application that integrates Retrieval-Augmented Generation (RAG) capabilities. The application allows users to:

- Ingest documents and generate embeddings
- Store embeddings in a PostgreSQL database
- Retrieve document embeddings for Q&A
- Generate answers based on retrieved embeddings using a pre-trained LLM model

## Features
- **Document Ingestion API:** Accepts and processes document data, generating embeddings.
- **Q&A API:** Accepts user queries and retrieves relevant document embeddings for RAG-based answer generation.
- **Document Selection API:** Allows users to specify which documents should be considered in the RAG-based Q&A process.
- **Uses `facebook/rag-sequence-nq` model** for retrieval-based answer generation.
- **Stores embeddings in PostgreSQL** for efficient retrieval.
- **Asynchronous API handling** for improved performance.

---

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PostgreSQL
- Pipenv or Virtualenv (optional but recommended)

### Step 1: Clone the Repository
```bash
git clone <repo-url>
cd document-rag-app
```

### Step 2: Create a Virtual Environment
Using `venv`:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Using `pipenv`:
```bash
pipenv shell
```

### Step 3: Install Dependencies
```bash
pip install transformers

pip install torch

pip install faiss-cpu

pip install datasets

pip install numpy

```

#### Required Packages:
```bash
pip install fastapi[all] torch transformers psycopg2
```

### Step 4: Set Up PostgreSQL Database
Create a PostgreSQL database and table:
```sql
CREATE DATABASE testdb;
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    embedding BYTEA
);
```

---

## Technologies Used
- **FastAPI** - Web framework for building APIs
- **Transformers (Hugging Face)** - Pre-trained LLMs for embeddings and RAG
- **PostgreSQL** - Storage for document embeddings
- **Torch** - Deep learning framework for model execution

---

## Notes
- Ensure PostgreSQL is running before starting the application.
- Modify database credentials in `db.py` before running.
- Consider optimizing retrieval algorithms for scalability.


