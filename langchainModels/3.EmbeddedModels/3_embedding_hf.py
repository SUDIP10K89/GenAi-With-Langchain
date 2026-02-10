from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = ["Hello world", "Hello world 2"]

vector = embedding.embed_documents(documents)

print(str(vector))
