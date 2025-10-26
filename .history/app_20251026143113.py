import streamlit as st
import torch
import numpy as np
from transformers import BertTokenizer, BertForTokenClassification
import json
import os
import pandas as pd # Kita tambahkan pandas untuk menampilkan tabel

st.write("--- SERVER SUDAH RESTART (VERSI BARU) ---") # <-- TANDA BAHWA FILE BARU BERJALAN

# --- KONFIGURASI ---
# Pastikan nama folder ini SAMA PERSIS dengan folder model Anda
MODEL_DIR = "./fine_tuned_bert_ner" 

# --- FUNGSI UNTUK MEMUAT MODEL (VERSI PERBAIKAN) ---
# @st.cache_resource akan menyimpan model di cache agar tidak di-load ulang
# Ini adalah fungsi yang sudah diperbaiki untuk membaca 'id2label' dari config
@st.cache_resource
def load_model_and_tokenizer(model_dir):
    """
    Memuat model, tokenizer, dan daftar tag dari direktori yang disimpan.
    """
    try:
        # Muat model dan tokenizer
        model = BertForTokenClassification.from_pretrained(model_dir)
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        
        # --- PERBAIKAN DARI ERROR SEBELUMNYA ---
        # Kita tidak lagi mencari 'tag_values.json'.
        # Sebagai gantinya, kita membaca 'id2label' dari file config.json model.
        # Ini dimuat secara otomatis ke dalam 'model.config'
        
        if not hasattr(model.config, 'id2label'):
            st.error("Error: 'id2label' tidak ditemukan di dalam config.json model.")
            return None, None, None, None

        # model.config.id2label adalah dictionary: {0: "O", 1: "B-indications", ...}
        # Kita ubah menjadi list: ["O", "B-indications", ...]
        # Ini penting agar kita bisa mapping output (angka) kembali ke label (teks)
        
        # --- INI ADALAH PERBAIKAN UNTUK KeyError: '0' ---
        # Mengubah str(i) menjadi i, karena keys-nya adalah integer
        tag_values = [model.config.id2label[i] for i in range(len(model.config.id2label))]
        
        # Tentukan device (GPU jika ada, jika tidak CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval() # Set model ke mode evaluasi (penting untuk prediksi)
        
        return model, tokenizer, tag_values, device
    
    except KeyError as e:
        # Tangani KeyError secara spesifik
        st.error(f"Error saat memuat model (KeyError): {e}")
        st.error("Ini biasanya terjadi jika 'id2label' di config.json tidak dimulai dari 0 atau key-nya bukan integer.")
        return None, None, None, None
    except Exception as e:
        # Tangani error umum lainnya
        st.error(f"Error saat memuat model: {e}")
        st.error(f"Pastikan folder '{model_dir}' ada di direktori yang sama dengan app.py")
        return None, None, None, None

# --- FUNGSI UNTUK PREDIKSI ---
def predict(text, model, tokenizer, tag_values, device):
    """
    Melakukan prediksi NER pada teks input.
    """
    # Tokenisasi teks input
    tokenized_sentence = tokenizer.encode(text, truncation=True, max_length=512)
    input_ids = torch.tensor([tokenized_sentence]).to(device)

    # Lakukan prediksi
    with torch.no_grad():
        output = model(input_ids)
    
    # Ambil label dengan skor tertinggi (argmax)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
    
    # Ubah ID token kembali menjadi token (kata)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    
    # --- Logika dari Notebook (menggabungkan token '##') ---
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        # Abaikan token spesial [CLS] dan [SEP]
        if token in ['[CLS]', '[SEP]']:
            continue
            
        if token.startswith("##"):
            # Jika token adalah BPE (sub-word), gabungkan dengan token sebelumnya
            if new_tokens:
                new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            # Jika token utuh, tambahkan token dan labelnya
            new_labels.append(tag_values[label_idx])
            new_tokens.append(token)
            
    return list(zip(new_tokens, new_labels))

# --- FUNGSI UNTUK TAMPILAN UI ---
def display_highlighted_text(results):
    """
    Menampilkan hasil sebagai teks yang di-highlight (UI Menarik)
    """
    # Definisikan warna untuk entitas. 
    # Anda bisa tambahkan jika punya banyak tipe entitas.
    # Di sini kita buat simpel: semua entitas akan berwarna biru muda.
    STYLE = """
        <mark style="
            background-color: #89CFF0; 
            padding: 2px 5px; 
            margin: 0px 3px;
            border-radius: 5px;
            border: 1px solid #008ECC;
        ">
            {token}
            <sub style="
                font-size: 0.7em; 
                opacity: 0.7; 
                margin-left: 3px;
                vertical-align: sub;
            ">
                {label}
            </sub>
        </mark>
    """
    
    html_output = '<div style="font-size: 1.1em; line-height: 2.0;">'
    
    for token, label in results:
        if label == "O":
            # Jika 'O' (Outside), tampilkan sebagai teks biasa
            html_output += f" {token}"
        else:
            # Jika entitas, tampilkan dengan highlight
            html_output += STYLE.format(token=token, label=label)
            
    html_output += "</div>"
    
    # --- INI ADALAH PERBAIKANNYA ---
    # Tambahkan unsafe_allow_html=True agar Streamlit merender HTML-nya
    st.markdown(html_output, unsafe_allow_html=True)

# --- FUNGSI UTAMA APLIKASI ---
def main():
    # Konfigurasi halaman
    st.set_page_config(
        page_title="Aplikasi NER Medis",
        page_icon="🧪",
        layout="wide"
    )

    # --- Header ---
    st.title("🧪 Aplikasi Named Entity Recognition (NER) dengan BERT")
    st.markdown("Aplikasi ini menggunakan model BERT yang di-fine-tune untuk mengenali entitas dari teks medis (berdasarkan notebook Anda).")

    # --- Memuat Model ---
    # Gunakan st.spinner agar terlihat loading saat model dimuat
    with st.spinner("Memuat model... Ini mungkin perlu beberapa saat..."):
        model, tokenizer, tag_values, device = load_model_and_tokenizer(MODEL_DIR)

    # Hanya lanjutkan jika model berhasil dimuat
    if model and tokenizer and tag_values and device:
        
        st.success("Model berhasil dimuat!")
        
        # --- Area Input ---
        st.header("Analisis Teks Anda")
        
        # Contoh teks diambil dari notebook Anda
        default_text = (
            "Pasteurellosis in japanese quail (Coturnix coturnix japonica) caused by Pasteurella multocida multocida A:4. \n\n"
            "Evaluation of transdermal penetration enhancers using a novel skin alternative. \n\n"
            "A novel alternative to animal skin models was developed in order to aid in the screening of transdermal penetration enhancer."
        )
        
        user_input = st.text_area("Masukkan teks untuk dianalisis di sini:", default_text, height=150)

        if st.button("🚀 Analisis Teks", type="primary"):
            if user_input:
                # Tampilkan spinner saat proses analisis
                with st.spinner("Menganalisis teks..."):
                    results = predict(user_input, model, tokenizer, tag_values, device)
                    
                    st.subheader("Hasil Analisis (Teks dengan Highlight)")
                    display_highlighted_text(results)
                    
                    # --- Tampilkan Data Mentah di dalam Expander ---
                    with st.expander("Lihat Data Mentah (Token & Tag)"):
                        # Filter hanya token yang BUKAN 'O' untuk data mentah
                        entities_only = [res for res in results if res[1] != 'O']
                        if entities_only:
                            df = pd.DataFrame(entities_only, columns=["Token", "Tag"])
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.info("Tidak ada entitas yang ditemukan.")
            else:
                st.warning("Silakan masukkan teks terlebih dahulu.")

# Menjalankan aplikasi
if __name__ == "__main__":
    main()

