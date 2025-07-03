# Simple preprocessing script for KazCausal corpus
import json

with open('data/kazcausal_corpus.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Loaded {len(data)} entries.")
