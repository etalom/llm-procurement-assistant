
import numpy as np

documents = open("documents.txt").read().split("\n")

def fake_embedding(text):
    return np.array([ord(c) for c in text[:10]])

doc_embeddings = [fake_embedding(d) for d in documents]

def similarity(a, b):
    return np.dot(a, b)

def search(query):
    q_emb = fake_embedding(query)
    scores = [similarity(q_emb, d) for d in doc_embeddings]
    best = np.argmax(scores)
    return documents[best]

if __name__ == "__main__":
    question = input("Ask a question: ")
    result = search(question)
    print("\nMost relevant document:")
    print(result)
