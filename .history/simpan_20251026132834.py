import os
import json

# Tentukan direktori untuk menyimpan model
OUTPUT_DIR = "./fine_tuned_bert_ner"

# Buat direktori jika belum ada
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Menyimpan model ke {OUTPUT_DIR}")

# Simpan model yang sudah di-fine-tune
model.save_pretrained(OUTPUT_DIR)

# Simpan tokenizer
tokenizer.save_pretrained(OUTPUT_DIR)

# Simpan tag2idx (penting untuk mapping output)
# (Ambil 'tag2idx' dari sel 8)
with open(os.path.join(OUTPUT_DIR, 'tag2idx.json'), 'w') as f:
    json.dump(tag2idx, f)

# (Ambil 'tag_values' dari sel 8)
with open(os.path.join(OUTPUT_DIR, 'tag_values.json'), 'w') as f:
    json.dump(tag_values, f)