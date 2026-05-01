from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

host = os.getenv("OPENSEARCH_HOST")
region = os.getenv("AWS_REGION")
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
        "host": host,
        "port": 443
    }],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

response = client.count(index="rag-index")

print(response)