from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Nepal is a landlocked country in South Asia, bordered by India and China.",
    "Kathmandu is the capital city of Nepal and the center of its political and cultural life.",
    "Machine learning enables systems to learn patterns from data without being explicitly programmed.",
    "LangChain is a framework designed to build applications powered by large language models.",
    "Vector databases are used to store embeddings for efficient similarity search.",
    "Climate change is causing rising temperatures, melting glaciers, and unpredictable weather patterns.",
    "Software engineering involves designing, developing, testing, and maintaining software systems.",
    "The MERN stack consists of MongoDB, Express.js, React, and Node.js."
]

query = "What is Langchain?"

doc_vector = embedding.embed_documents(documents)
query_vector = embedding.embed_query(query)

scores = cosine_similarity([query_vector],doc_vector)[0]

index,score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(f"Query: {query}")
print(f"Most similar document: {documents[index]} with score {score}")