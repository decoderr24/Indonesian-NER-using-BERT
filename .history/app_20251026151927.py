import os
import torch
import numpy as np
import pandas as pd
import streamlit as st
from transformers import BertTokenizer, BertForTokenClassification

# ======================================================================================
# KONFIGURASI MODEL
# ======================================================================================
# Model publik Hugging Face
MODEL_DIR = "Decoder24/indonesian-ner-bert"

# Nonaktifkan file watcher agar tidak error di Streamlit Cloud
os.environ["STREAMLIT_WATCH_FILESYSTEM"] = "false"

# ======================================================================================
# FUNGSI PEMUATAN MODEL
# ======================================================================================
@st.cache_resource
def load_model_and_tokenizer(model_repo: str):
    """
    Memuat model & tokenizer dari Hugging Face Hub.
    """
    try:
        # Deteksi apakah input adalah repo HF atau path lokal
        if os.path.isdir(model_repo):
            st.info(f"Memuat model dari folder lokal: {model_repo}")
            model = BertForTokenClassification.from_pretrained(model_repo)
            tokenizer = BertTokenizer.from_pretrained(model_repo)
        else:
            st.info(f"Memuat model dari Hugging Face Hub: {model_repo}")
            model = BertForTokenClassification.from_pretrained(model_repo, use_auth_token=False)
            tokenizer = BertTokenizer.from_pretrained(model_repo, use_auth_token=False)

        # Validasi konfigurasi label
        if not hasattr(model.config, "id2label"):
            st.error("Config model tidak berisi 'id2label'. Pastikan model BERT-nya sudah di-fine-tune untuk NER.")
            return None, None, None, None

        tag_values = [model.config.id2label[i] for i in range(len(model.config.id2label))]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return model, tokenizer, tag_values, device

    except Exception as e:
        st.error(f"❌ Error saat memuat model: {e}")
        st.error(f"Pastikan model '{model_repo}' dapat diakses secara publik di Hugging Face Hub.")
        return None, None, None, None

# ======================================================================================
# FUNGSI PREDIKSI NER
# ======================================================================================
def predict(text, model, tokenizer, tag_values, device):
    """
    Melakukan prediksi Named Entity Recognition (NER) pada teks input.
    """
    tokenized_sentence = tokenizer.encode(text, truncation=True, max_length=512)
    input_ids = torch.tensor([tokenized_sentence]).to(device)

    with torch.no_grad():
        output = model(input_ids)

    label_indices = np.argmax(output[0].to("cpu").numpy(), axis=2)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to("cpu").numpy()[0])

    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token in ["[CLS]", "[SEP]"]:
            continue
        if token.startswith("##") and new_tokens:
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_tokens.append(token)
            new_labels.append(tag_values[label_idx])

    return list(zip(new_tokens, new_labels))

# ======================================================================================
# ANTARMUKA STREAMLIT
# ======================================================================================
def main():
    # Konfigurasi tampilan
    st.set_page_config(page_title="Aplikasi NER Medis", page_icon="🧬", layout="wide")

    # Header
    st.title("🧪 Aplikasi Named Entity Recognition (NER) dengan BERT")
    st.markdown("Aplikasi ini menggunakan model **BERT** yang telah di-*fine-tune* untuk mengenali entitas dalam teks medis bahasa Inggris maupun Indonesia.")
    st.markdown("---")

    # Load model
    with st.spinner("⏳ Memuat model dari Hugging Face..."):
        model, tokenizer, tag_values, device = load_model_and_tokenizer(MODEL_DIR)

    if not all([model, tokenizer, tag_values, device]):
        st.stop()

    st.success("✅ Model berhasil dimuat!")

    # Area input teks
    st.header("🩺 Analisis Teks Medis")
    default_text = (
        "Pasteurellosis in japanese quail (Coturnix coturnix japonica) caused by Pasteurella multocida multocida A:4. "
        "Evaluation of transdermal penetration enhancers using a novel skin alternative. "
        "A novel alternative to animal skin models was developed in order to aid in the screening of transdermal penetration enhancer."
    )

    user_input = st.text_area("Masukkan teks untuk dianalisis:", default_text, height=150)

    if st.button("🚀 Analisis Sekarang"):
        if not user_input.strip():
            st.warning("Masukkan teks terlebih dahulu!")
            st.stop()

        with st.spinner("🔍 Menganalisis entitas dalam teks..."):
            results = predict(user_input, model, tokenizer, tag_values, device)

            st.subheader("📋 Hasil Analisis (Token & Tag)")
            df = pd.DataFrame(results, columns=["Token", "Tag"])
            st.dataframe(df, use_container_width=True)

            st.subheader("💡 Entitas yang Terdeteksi")
            entities_only = df[df["Tag"] != "O"]
            if not entities_only.empty:
                st.dataframe(entities_only, use_container_width=True)
            else:
                st.info("Tidak ada entitas yang terdeteksi dalam teks ini.")

    # Footer
    st.markdown("---")
    st.caption("Developed by Decoder24 • Model: [Decoder24/indonesian-ner-bert](https://huggingface.co/Decoder24/indonesian-ner-bert)")

# ======================================================================================
# ENTRY POINT
# ======================================================================================
if __name__ == "__main__":
    main()
