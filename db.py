import psycopg2

# Connect to the PostgreSQL database
conn = psycopg2.connect(database="testdb", user="postgres", password="passdb", host="127.0.0.1", port="5432")
cur = conn.cursor()

def store_embeddings(embeddings):
    cur.execute("INSERT INTO embeddings (embedding) VALUES (%s)", (embeddings.detach().numpy(),))
    conn.commit()

def retrieve_embeddings(documents):
    cur.execute("SELECT embedding FROM embeddings WHERE document IN %s", (documents,))
    embeddings = cur.fetchall()
    return torch.tensor(embeddings)