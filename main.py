from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("⏳ Loading model...")

start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu"
)

print(f"✅ Model loaded in {round(time.time() - start_time, 2)}s")

# ⚡ Warm-up (removes first lag spike)
print("🔥 Warming up...")
warm_inputs = tokenizer("Hi", return_tensors="pt")
_ = model.generate(**warm_inputs, max_new_tokens=1)

print("⚡ Ready!\n")

# 🔥 Your actual prompt
prompt = "Write a fast Python function to reverse a string."

inputs = tokenizer(prompt, return_tensors="pt")

start_time = time.time()

outputs = model.generate(
    **inputs,
    max_new_tokens=80,      # ⚡ keep this low for speed
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("🧠 AI Response:\n")
print(response)

print(f"\n⚡ Generated in {round(time.time() - start_time, 2)}s")
