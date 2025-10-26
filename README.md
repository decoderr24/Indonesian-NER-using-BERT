
 # Indonesia Named Entity Recognition (NER) using BERT

Aplikasi berbasis **Streamlit** untuk mendeteksi entitas bernama (*Named Entity Recognition / NER*) pada teks berbahasa Indonesia menggunakan **BERT (Bidirectional Encoder Representations from Transformers)** yang telah di-*fine-tune*.  
Proyek ini dirancang untuk kebutuhan analisis teks domain medis, namun dapat dikembangkan untuk domain lain seperti berita, hukum, atau sosial media.
<img width="2880" height="2216" alt="localhost_8501_ (4)" src="https://github.com/user-attachments/assets/b5da265f-d07c-4aeb-9f56-ce377fc8a985" />

---

## 🚀 **Fitur Utama**
- 🔍 **Prediksi otomatis entitas** (mis. nama penyakit, spesies, lokasi, dsb.) dari teks input.
- 🎨 **Highlight visual interaktif** untuk setiap entitas yang terdeteksi.
- ⚙️ **Berbasis model BERT yang telah di-fine-tune** untuk tugas token classification.
- 📊 **Tabel hasil entitas** yang dapat diperluas (expandable).
- 💻 **Aplikasi berbasis web (Streamlit)** — berjalan lokal maupun di-deploy ke cloud.

---

## 🧠 **Model yang Digunakan**
Model menggunakan arsitektur **BERT (Bidirectional Encoder Representations from Transformers)** yang telah di-*fine-tune* pada dataset NER Bahasa Indonesia.  
Struktur folder model:

fine_tuned_bert_ner/
│
├── config.json

├── pytorch_model.bin

├── tokenizer_config.json

├── vocab.txt

└── special_tokens_map.json


Pastikan folder ini berada **satu direktori** dengan file `app.py`.

---

## 🛠️ **Cara Menjalankan Proyek**

### 1️⃣ Clone Repository
```bash
git clone https://github.com/decoderr24/Indonesian-NER-using-BERT.git
cd Indonesian-NER-using-BERT
```
2️⃣ Install Dependencies

Gunakan Python 3.8+ dan jalankan:

```bash
pip install -r requirements.txt

```


Atau manual:
```bash
pip install streamlit torch transformers pandas
```
3️⃣ Jalankan Aplikasi
```bash
streamlit run app.py
```
Kemudian buka browser di:


http://localhost:8501


