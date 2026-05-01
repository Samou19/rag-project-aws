import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"

st.title("🤖 RAG Assistant AWS")

question = st.text_input("Pose ta question")

if st.button("Envoyer"):

    response = requests.post(API_URL, json={"question": question})

    if response.status_code == 200:
        data = response.json()

        st.subheader("🧠 Réponse")
        st.write(data["answer"])

        st.subheader("📚 Contexte")
        st.write(data["context"])
    else:
        st.error("Erreur API")