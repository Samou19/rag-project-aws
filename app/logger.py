import json
from datetime import datetime

LOG_FILE = "logs.json"

def log_query(question, answer):

    entry = {
        "timestamp": str(datetime.now()),
        "question": question,
        "answer": answer
    }

    try:
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
    except:
        data = []

    data.append(entry)

    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)