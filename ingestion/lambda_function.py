import boto3
import json
import os
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import zipfile
import xml.etree.ElementTree as ET
import io
import PyPDF2

# 🔹 AWS config
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

# 🔹 OpenSearch (IMPORTANT: sans https)
host = "search-rag-opensearch-bppgxgm5gtv5dbnk3vh5yh3h5u.eu-west-3.es.amazonaws.com"

client = OpenSearch(
    hosts=[{"host": host, "port": 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

# 🔹 AWS clients
s3 = boto3.client("s3")
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# 🔹 Chunking
def chunk_text(text, size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i + size])
    return chunks

# 🔹 Embedding
def get_embedding(text):
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        body=json.dumps({"inputText": text})
    )
    result = json.loads(response["body"].read())
    return result["embedding"]

# 🔹 Extraction texte (multi-format)
def extract_text(file_bytes, key):

    # 📄 TXT
    if key.endswith(".txt"):
        return file_bytes.decode("utf-8")

    # 📄 PDF
    elif key.endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    # 📄 DOCX (SANS python-docx 🔥)
    elif key.endswith(".docx"):
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as docx:
            xml_content = docx.read("word/document.xml")

        tree = ET.fromstring(xml_content)

        text = []
        for elem in tree.iter():
            if elem.text:
                text.append(elem.text)

        return " ".join(text)

    else:
        return ""

# 🔹 Lambda handler
def lambda_handler(event, context):

    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = event["Records"][0]["s3"]["object"]["key"]

    print(f"Processing file: {key}")

    # 🔹 lecture fichier
    response = s3.get_object(Bucket=bucket, Key=key)
    file_bytes = response["Body"].read()

    # 🔹 extraction texte
    text = extract_text(file_bytes, key)

    # 🔹 chunking
    chunks = chunk_text(text)
    print(f"{len(chunks)} chunks créés")

    # 🔹 indexation
    for i, chunk in enumerate(chunks):

        emb = get_embedding(chunk)

        doc = {
            "text": chunk,
            "embedding": emb,
            "source": key
        }

        client.index(
            index="rag-index",
            id=f"{key}_{i}",   # 🔥 important
            body=doc
        )

    print("Indexation terminée")

    return {"statusCode": 200}