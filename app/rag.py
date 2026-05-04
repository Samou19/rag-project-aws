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
def rerank(query, docs):
    return sorted(
        docs,
        key=lambda d: (
            sum(word in d.lower() for word in query.lower().split()),
            -len(d)
        ),
        reverse=True
    )

def search_docs(query, k=2):

    query_vector = get_embedding(query)

    search_query = {
        "size": k,
        "query": {
            "bool": {
                "should": [
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_vector,
                                "k": k
                            }
                        }
                    },
                    {
                        "match": {
                            "text": {
                                "query": query,
                                "boost": 0.3
                            }
                        }
                    }
                ]
            }
        }
    }

    response = client.search(index="rag-index", body=search_query)
    hits = response["hits"]["hits"]

    docs = [hit["_source"]["text"] for hit in hits]

    # 🔥 filtrage sémantique
    docs = [
        d for d in docs
        if any(word in d.lower() for word in query.lower().split())
    ]

    # 🔥 reranking
    docs = rerank(query, docs)

    return "\n\n".join(docs[:2])


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
                                "text":  f"""
Tu es un assistant expert en conformité bancaire.

Réponds uniquement avec les informations du contexte.

Instructions :
- Réponse claire et concise
- Appuie ta réponse sur les éléments du contexte
- N'invente rien
- Si l'information est absente → "Je ne sais pas"

Contexte:
{context}

Question:
{question}

Réponse:
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