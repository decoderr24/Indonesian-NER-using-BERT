import streamlit as st
import torch
import numpy as np
from transformers import BertTokenizer, BertForTokenClassification
import json
import os # <-- Pastikan 'os' di-import
import pandas as pd 

# --- KONFIGURASI ---
MODEL_DIR = "Decoder24/indonesian-ner-bert"


# --- FUNGSI UNTUK MEMUAT MODEL ---
@st.cache_resource
def load_model_and_tokenizer(model_dir_relative): # <-- ganti nama argumen
    """
    Memuat model, tokenizer, dan daftar tag dari direktori yang disimpan.
    """
    try:
        # --- PERBAIKAN UNTUK STREAMLIT CLOUD ---
        # Ubah path relatif (misal: "./fine_tuned_bert_ner")
        # menjadi path absolut (misal: "/mount/src/.../fine_tuned_bert_ner")
        # Ini mencegah transformers salah mengira path lokal sebagai ID repo Hugging Face
        model_dir_absolute = os.path.abspath(model_dir_relative)
        
        # Muat model & tokenizer dari path absolut
        model = BertForTokenClassification.from_pretrained(model_dir_absolute)
        tokenizer = BertTokenizer.from_pretrained(model_dir_absolute)
        # --- AKHIR PERBAIKAN ---

        if not hasattr(model.config, 'id2label'):
            st.error("Error: 'id2label' tidak ditemukan di dalam config.json model.")
            return None, None, None, None

        tag_values = [model.config.id2label[i] for i in range(len(model.config.id2label))]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        return model, tokenizer, tag_values, device
    
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        st.error(f"Pastikan folder '{model_dir_relative}' ada di direktori yang sama dengan app.py")
        return None, None, None, None

# --- FUNGSI UNTUK PREDIKSI ---
def predict(text, model, tokenizer, tag_values, device):
    """
    Melakukan prediksi NER pada teks input.
    """
    tokenized_sentence = tokenizer.encode(text, truncation=True, max_length=512)
    input_ids = torch.tensor([tokenized_sentence]).to(device)

    with torch.no_grad():
        output = model(input_ids)
    
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token in ['[CLS]', '[SEP]']:
            continue
            
        if token.startswith("##"):
            if new_tokens:
                new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(tag_values[label_idx])
            new_tokens.append(token)
            
    return list(zip(new_tokens, new_labels))

# --- FUNGSI UTAMA APLIKASI ---
def main():
    st.set_page_config(
        page_title="Aplikasi NER Medis",
        page_icon="🧪",
        layout="wide"
    )

    st.title("🧪 Aplikasi Named Entity Recognition (NER) dengan BERT")
    st.markdown("Aplikasi ini menggunakan model BERT yang di-fine-tune untuk mengenali entitas dari teks medis.")

    with st.spinner("Memuat model..."):
        # Panggil fungsi dengan MODEL_DIR global
        model, tokenizer, tag_values, device = load_model_and_tokenizer(MODEL_DIR)

    if model and tokenizer and tag_values and device:
        
        st.success("Model berhasil dimuat!")
        
        st.header("Analisis Teks Anda")
        
        default_text = (
            "Pasteurellosis in japanese quail (Coturnix coturnix japonica) caused by Pasteurella multocida multocida A:4. \n\n"
            "Evaluation of transdermal penetration enhancers using a novel skin alternative. \n\n"
            "A novel alternative to animal skin models was developed in order to aid in the screening of transdermal penetration enhancer."
        )
        
        user_input = st.text_area("Masukkan teks untuk dianalisis di sini:", default_text, height=150)

        if st.button("🚀 Analisis Teks", type="primary"):
            if user_input:
                with st.spinner("Menganalisis teks..."):
                    results = predict(user_input, model, tokenizer, tag_values, device)
                    
                    st.subheader("Hasil Analisis (Tabel Data)")
                    
                    df = pd.DataFrame(results, columns=["Token", "Tag"])
                    st.dataframe(df, use_container_width=True)
                    
                    with st.expander("Lihat Entitas yang Ditemukan Saja"):
                        entities_only = df[df["Tag"] != 'O']
                        if not entities_only.empty:
                            st.dataframe(entities_only, use_container_width=True)
                        else:
                            st.info("Tidak ada entitas yang ditemukan.")
            else:
                st.warning("Silakan masukkan teks terlebih dahulu.")

# Menjalankan aplikasi
if __name__ == "__main__":
    main()

