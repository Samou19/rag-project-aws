from app.rag import search_docs, generate_answer
docs = search_docs("KYC")
print(docs)