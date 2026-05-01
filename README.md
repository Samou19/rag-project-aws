# RAG Assistant AWS

## Description
Assistant intelligent basé sur RAG utilisant :
- Amazon Bedrock
- OpenSearch
- FastAPI

## Architecture
User → API → Retrieval → LLM → Response

## Installation
pip install -r requirements.txt

## Run
uvicorn app.main:app --reload

## Demo
http://<IP>:8000/docs