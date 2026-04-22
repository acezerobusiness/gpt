import os
import time
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
REPO = os.environ["REPO"]

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("⏳ Loading model (only once)...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print("✅ Model ready. Worker started.\n")

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        temperature=0.7,
        top_p=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_issues():
    url = f"https://api.github.com/repos/{REPO}/issues"
    r = requests.get(url, headers=HEADERS)
    return r.json()

def comment(issue, text):
    requests.post(issue["comments_url"], headers=HEADERS, json={"body": text})

def mark_done(issue):
    # add label so we don’t process again
    url = issue["url"] + "/labels"
    requests.post(url, headers=HEADERS, json={"labels": ["done"]})

def is_processed(issue):
    labels = [l["name"] for l in issue.get("labels", [])]
    return "done" in labels

# 🔁 MAIN LOOP
start_time = time.time()

while True:
    print("🔎 Checking for new requests...")

    issues = get_issues()

    for issue in issues:
        if is_processed(issue):
            continue

        prompt = issue["body"]
        issue_number = issue["number"]

        print(f"⚡ Processing issue #{issue_number}")

        try:
            response = generate(prompt)

            comment(issue, response)
            mark_done(issue)

            print(f"✅ Done #{issue_number}")

        except Exception as e:
            comment(issue, f"Error: {str(e)}")

    # ⏱️ stop after ~5.5 hours (safe margin)
    if time.time() - start_time > 5.5 * 3600:
        print("⏹️ Stopping before timeout")
        break

    time.sleep(30)  # 🔥 important (avoid rate limits)
