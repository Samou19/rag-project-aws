import json
from app.rag import search_docs, generate_answer
from datasets import Dataset
from ragas import evaluate
from langchain_aws import ChatBedrock, BedrockEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


# Embeddings Bedrock
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    region_name="us-east-1"
)

ragas_embeddings = LangchainEmbeddingsWrapper(bedrock_embeddings)

# 🔹 LLM Bedrock (juge)
bedrock_llm = ChatBedrock(
    model_id="amazon.nova-lite-v1:0",
    region_name="us-east-1"
)

ragas_llm = LangchainLLMWrapper(bedrock_llm)

# métriques RAGAS
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# 🔹 tes fonctions RAG

# =========================
# DATASET DE TEST
# =========================

test_data = [
    {
        "question": "Qu'est-ce que le KYC ?",
        "ground_truth": "Le KYC est un processus permettant de vérifier l'identité des clients, incluant la collecte de documents et l'analyse du risque."
    },
    {
        "question": "Que comprend le KYC ?",
        "ground_truth": "Le KYC comprend la vérification d'identité, la collecte de documents et l'analyse du risque client."
    },
    {
        "question": "Quand une transaction est-elle considérée comme suspecte ?",
        "ground_truth": "Une transaction est suspecte si le montant dépasse 5000€, si le pays est différent ou s'il y a plus de 5 transactions en 24h."
    },
    {
        "question": "Quelles actions sont prises en cas de fraude ?",
        "ground_truth": "Les actions incluent le blocage de la transaction, la notification de la conformité et la vérification du client."
    },
    {
        "question": "Que comprend la conformité bancaire ?",
        "ground_truth": "La conformité bancaire inclut l'AML, le KYC et la surveillance des transactions."
    }
]


# =========================
# PIPELINE RAG
# =========================

def run_rag(dataset):
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in dataset:
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\n🔎 Question: {question}")

        # 🔹 Retrieval
        context = search_docs(question)

        # 🔹 Generation
        answer = generate_answer(question, context)

        print(f"📚 Context:\n{context[:200]}...")
        print(f"🤖 Answer: {answer}")

        questions.append(question)
        answers.append(answer)
        contexts.append([context])  # ⚠️ liste obligatoire pour RAGAS
        ground_truths.append(ground_truth)

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })


# =========================
# ÉVALUATION
# =========================

def evaluate_rag():
    dataset = run_rag(test_data)

    print("\n🚀 Lancement évaluation RAGAS...")

    results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    llm=ragas_llm,
    embeddings=ragas_embeddings   # 🔥 AJOUT CRITIQUE
    )

    print("\n📊 Résultats :")
    print(results)

    # 🔹 sauvegarde
    with open("rag_evaluation.json", "w") as f:
        json.dump(results.to_pandas().to_dict(), f, indent=2)

    print("\n💾 Résultats sauvegardés dans rag_evaluation.json")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    evaluate_rag()