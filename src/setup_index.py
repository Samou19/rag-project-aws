import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

region = "eu-west-3"
service = "es"

credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    service,
    session_token=credentials.token
)

host = "search-rag-opensearch-bppgxgm5gtv5dbnk3vh5yh3h5u.aos.eu-west-3.on.aws"

client = OpenSearch(
    hosts=[{"host": host, "port": 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

# ✅ DELETE
if client.indices.exists(index="rag-index"):
    client.indices.delete(index="rag-index")
    print("Index supprimé")

# ✅ CREATE
client.indices.create(
    index="rag-index",
    body={
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 1536
                },
                "source": {"type": "keyword"}
            }
        }
    }
)

print("Index recréé avec knn_vector ✅")