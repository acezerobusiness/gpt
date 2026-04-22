import os
import time
import requests
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

# 🔥 Force real-time logs (fixes "stuck" issue)
sys.stdout.reconfigure(line_buffering=True)

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


# 🧠 Generate response
def generate(prompt):
    print("🧠 Generating...")

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        max_length=None  # removes warning
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 📥 Fetch issues
def get_issues():
    url = f"https://api.github.com/repos/{REPO}/issues"
    r = requests.get(url, headers=HEADERS)

    print(f"🌐 GitHub API status: {r.status_code}")

    if r.status_code != 200:
        print("❌ API error:", r.text)
        return []

    return r.json()


# 💬 Comment result
def comment(issue, text):
    print(f"💬 Commenting on #{issue['number']}")

    requests.post(
        issue["comments_url"],
        headers=HEADERS,
        json={"body": text}
    )


# 🏷️ Mark issue as done
def mark_done(issue):
    print(f"🏷️ Marking #{issue['number']} as done")

    url = issue["url"] + "/labels"

    requests.post(
        url,
        headers=HEADERS,
        json={"labels": ["done"]}
    )


# 🔍 Check if processed
def is_processed(issue):
    labels = [l["name"] for l in issue.get("labels", [])]
    print(f"🔎 Issue #{issue['number']} labels:", labels)

    return "done" in labels


# ⏱️ MAIN LOOP
start_time = time.time()

while True:
    print("\n🔁 Checking for new issues...")

    issues = get_issues()
    print(f"📦 Found {len(issues)} issues")

    for issue in issues:
        issue_number = issue["number"]

        print(f"\n➡️ Checking issue #{issue_number}")

        if is_processed(issue):
            print("⏭️ Skipping (already processed)")
            continue

        # 🔥 FIX: handle None safely
        prompt = issue.get("body") or ""

        if not prompt.strip():
            print("⚠️ Empty or None prompt")

            comment(issue, "⚠️ Empty prompt. Please send something.")
            mark_done(issue)
            continue

        try:
            print(f"⚡ Processing issue #{issue_number}")

            response = generate(prompt)

            comment(issue, response)
            mark_done(issue)

            print(f"✅ Completed #{issue_number}")

        except Exception as e:
            print("❌ Error:", str(e))

            comment(issue, f"Error: {str(e)}")
            mark_done(issue)

    # ⏱️ stop before GitHub kills job
    if time.time() - start_time > 5.5 * 3600:
        print("⏹️ Stopping before timeout")
        break

    print("💓 Worker alive... sleeping 30s\n")
    time.sleep(30)
