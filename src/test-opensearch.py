from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3

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

client = OpenSearch(
    hosts=[{
        "host": "search-rag-opensearch-bppgxgm5gtv5dbnk3vh5yh3h5u.eu-west-3.es.amazonaws.com",
        "port": 443
    }],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

response = client.count(index="rag-index")

print(response)