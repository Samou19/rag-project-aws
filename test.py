from app.rag import search_docs, generate_answer
docs = search_docs("Quelles actions sont prises en cas de fraude ?")
print(docs)