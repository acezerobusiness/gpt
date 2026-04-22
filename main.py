import os
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
REPO = os.environ["GITHUB_REPOSITORY"]

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        temperature=0.7,
        top_p=0.9
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def comment(issue_number, text):
    url = f"https://api.github.com/repos/{REPO}/issues/{issue_number}/comments"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}

    requests.post(url, headers=headers, json={"body": text})

def main():
    issue_number = os.environ["ISSUE_NUMBER"]
    prompt = os.environ["ISSUE_BODY"]

    print("Generating response...")
    response = generate(prompt)

    comment(issue_number, response)

if __name__ == "__main__":
    main()
