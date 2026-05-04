import boto3
import json
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# CONFIG
# =========================
region = os.getenv("AWS_REGION")
service = "es"

host = os.getenv("OPENSEARCH_HOST")

# =========================
# AUTH AWS
# =========================
session = boto3.Session()
credentials = session.get_credentials()

if credentials is None:
    raise ValueError("AWS credentials not found")

awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    service,
    session_token=credentials.token
)

# =========================
# CLIENT OPENSEARCH
# =========================
client = OpenSearch(
    hosts=[{"host": host, "port": 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

# =========================
# CLIENT BEDROCK (⚠️ US EAST)
# =========================
bedrock = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"  # 🔥 IMPORTANT
)

# =========================
# EMBEDDING
# =========================
def get_embedding(text):
    try:
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v1",
            body=json.dumps({"inputText": text})
        )
        result = json.loads(response["body"].read())
        return result["embedding"]
    except Exception as e:
        print("❌ Erreur embedding :", e)
        return None


# =========================
# SEARCH (vector + fallback)
# =========================
def search_docs(query):

    query_vector = get_embedding(query)

    if query_vector is None:
        return ""

    try:
        # 🔥 recherche vectorielle
        search_query = {
            "size": 3,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": 3
                    }
                }
            }
        }

        response = client.search(index="rag-index", body=search_query)

    except Exception as e:
        print("⚠️ KNN failed → fallback BM25:", e)

        # 🔁 fallback texte classique
        search_query = {
            "size": 3,
            "query": {
                "match": {
                    "text": query
                }
            }
        }

        response = client.search(index="rag-index", body=search_query)

    results = [hit["_source"]["text"] for hit in response["hits"]["hits"]]

    return "\n".join(results)


# =========================
# GENERATION (SAFE MODE)
# =========================
def generate_answer(question, context):

    try:
        response = bedrock.invoke_model(
            modelId="amazon.nova-lite-v1:0",
            body=json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": f"""
Réponds à la question avec le contexte.
Si tu ne sais pas, dis "Je ne sais pas".

Contexte:
{context}

Question:
{question}
"""
                            }
                        ]
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": 300,
                    "temperature": 0.3
                }
            })
        )

        result = json.loads(response["body"].read())

        return result["output"]["message"]["content"][0]["text"]

    except Exception as e:
        print("❌ Nova failed:", e)
        return f"Fallback:\n{context[:500]}"

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    question = input("🔎 Pose ta question: ")

    context = search_docs(question)

    print("\n📚 Contexte trouvé:\n", context)

    answer = generate_answer(question, context)

    print("\n🤖 Réponse:\n", answer)