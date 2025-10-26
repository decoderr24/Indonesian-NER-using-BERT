import streamlit as st
import torch
import numpy as np
from transformers import BertTokenizer, BertForTokenClassification
import json
import os

# Tentukan path ke model yang sudah disimpan
MODEL_DIR = "./ner_bert_model"

# --- Fungsi untuk Memuat Model dan Tokenizer ---
# @st.cache_resource akan menyimpan model di cache agar tidak di-load ulang
@st.cache_resource
def load_model_and_tokenizer(model_dir):
    try:
        model = BertForTokenClassification.from_pretrained(model_dir)
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        
        # Muat tag_values
        with open(os.path.join(model_dir, 'tag_values.json'), 'r') as f:
            tag_values = json.load(f)
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval() # Set model ke mode evaluasi
        
        return model, tokenizer, tag_values, device
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None, None, None, None

# --- Fungsi untuk Prediksi ---
def predict(text, model, tokenizer, tag_values, device):
    tokenized_sentence = tokenizer.encode(text)
    input_ids = torch.tensor([tokenized_sentence]).to(device)

    with torch.no_grad():
        output = model(input_ids)
    
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    
    # Logika dari sel 36 (menggabungkan token BPE '##')
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(tag_values[label_idx])
            new_tokens.append(token)
            
    # Menggabungkan token dan label
    results = []
    for token, label in zip(new_tokens, new_labels):
        # Abaikan token spesial [CLS] dan [SEP]
        if token not in ['[CLS]', '[SEP]']:
            results.append((token, label))
    return results

# --- Setup UI Streamlit ---
st.set_page_config(layout="wide")
st.title("🧪 Aplikasi Named Entity Recognition (NER) dengan BERT")
st.write("Aplikasi ini menggunakan model BERT yang di-fine-tune untuk mengenali entitas dari teks medis.")

# Muat model
model, tokenizer, tag_values, device = load_model_and_tokenizer(MODEL_DIR)

if model:
    # Ambil contoh teks dari notebook Anda
    default_text = """
Pasteurellosis in japanese quail (Coturnix coturnix japonica) caused by Pasteurella multocida multocida A:4. 
Evaluation of transdermal penetration enhancers using a novel skin alternative.
    """
    
    # Buat Text Area untuk input pengguna
    user_input = st.text_area("Masukkan teks untuk dianalisis:", default_text, height=150)

    if st.button("🚀 Analisis Teks"):
        if user_input:
            with st.spinner("Menganalisis..."):
                results = predict(user_input, model, tokenizer, tag_values, device)
                
                st.subheader("Hasil Analisis:")
                
                # Menampilkan hasil dengan styling
                # (Ini adalah cara sederhana, bisa juga pakai st.dataframe)
                
                # Kita buat 2 kolom agar lebih rapi
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Token**")
                    for token, label in results:
                        st.write(token)
                
                with col2:
                    st.markdown("**Tag (Entitas)**")
                    for token, label in results:
                        if label == "O":
                            st.write(label)
                        else:
                            # Beri tanda jika bukan 'O'
                            st.success(f"**{label}**") 
        else:
            st.warning("Silakan masukkan teks terlebih dahulu.")
else:
    st.error("Model tidak dapat dimuat. Pastikan folder `ner_bert_model` ada di direktori yang sama dengan `app.py`.")