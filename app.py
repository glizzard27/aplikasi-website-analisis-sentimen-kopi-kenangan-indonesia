import streamlit as st
import torch
import pickle
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer
from collections import Counter
from wordcloud import WordCloud

# Fungsi-fungsi preprocessing
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[@#]\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_text(text, kamus_normalisasi):
    words = text.split()
    normalized_words = [kamus_normalisasi.get(word, word) for word in words]
    return ' '.join(normalized_words)

# Memuat kamus normalisasi sekali saja saat aplikasi dimulai
try:
    df_kamus = pd.read_excel('kamus_normalisasi.xlsx', engine='openpyxl')
    normalization_dict = dict(zip(df_kamus['tidak_baku'], df_kamus['kata_baku']))
except Exception as e:
    normalization_dict = {}
    st.warning(f"Gagal memuat kamus normalisasi: {e}")


def full_preprocess(text):
    cleaned_text = clean_text(text)
    normalized_text = normalize_text(cleaned_text, normalization_dict)
    return normalized_text

# Fungsi untuk memuat model 
@st.cache_resource
def load_model_assets():
    model_path = "model_kopikenangan_terbaik"
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        label_encoder_path = os.path.join(model_path, "label_encoder.pkl")
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        print("Aset model berhasil dimuat.")
        return model, tokenizer, label_encoder
    except Exception as e:
        st.error(f"Gagal memuat aset model. Pastikan folder 'model_kopikenangan_terbaik' ada di direktori yang sama. Error: {e}")
        return None, None, None

# Fungsi untuk prediksi sentimen
def predict_sentiment(text, model, tokenizer, label_encoder, device):
    if model is None: return "Error: Model tidak dimuat"
    
    processed_text = full_preprocess(text)
    inputs = tokenizer(processed_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)[0]
        predicted_class_id = torch.argmax(probabilities).item()
        predicted_label = label_encoder.classes_[predicted_class_id]
        all_probabilities = {label: prob.item() for label, prob in zip(label_encoder.classes_, probabilities)}
    return predicted_label, all_probabilities

# Fungsi untuk memuat contoh ulasan dari test dataset
@st.cache_data
def load_sample_reviews(file_path, column_name):
    try:
        df_samples = pd.read_csv(file_path)
        return df_samples[column_name].tolist()
    except FileNotFoundError:
        st.error(f"File '{file_path}' tidak ditemukan. Fitur contoh acak tidak akan berfungsi.")
        return []
    except Exception as e:
        st.error(f"Gagal memuat file contoh: {e}")
        return []

# BAGIAN UTAMA APLIKASI WEBSITE ANALISIS SENTIMEN ULASAN KOPI KENANGAN INDONESIA
# Mengatur konfigurasi halaman
st.set_page_config(
    page_title="Aplikasi Website Analisis Sentimen Ulasan Kopi Kenangan Indonesia",
    page_icon= "☕",
    layout="wide"
)
# Memuat aset sekali di awal
model, tokenizer, label_encoder = load_model_assets()
processed_samples = load_sample_reviews('contoh_ulasan_dari_test_dataset.csv', 'content_preprocessed')
raw_samples = load_sample_reviews('contoh_ulasan_mentah.csv', 'content')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model:
    model.to(device)

# Menginisialisasi Session State untuk semua input dengan string kosong.
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'batch_input' not in st.session_state:
    st.session_state.batch_input = ""
if 'simulator_input' not in st.session_state:
    st.session_state.simulator_input = ""

# Sidebar untuk Navigasi
with st.sidebar:
    st.image("logo_kopi_kenangan.png", width=200,)
    st.title("Aplikasi Website Analisis Sentimen Ulasan Kopi Kenangan Indonesia")
    st.markdown("---")
    page = st.radio(
        "Pilih Halaman:",
        ("Dashboard", "Simulator Preprocessing Teks Ulasan", "Analisis Sentimen Tunggal", "Analisis Sentimen Batch"),
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.write("Muhammad Fajar Baihaqi\n50421950")

# Konten di Dashboard
if page == "Dashboard":
    st.title("📊 Dashboard Informasi Proyek Analisis Sentimen Kopi Kenangan Indonesia")
    st.write("Selamat datang di Dashboard utama Aplikasi Website Analisis Sentimen Ulasan Kopi Kenangan Indonesia.")
    st.info("Halaman ini berisi Informasi dan Statistik Dataset beserta Informasi Model. Gunakan menu di sidebar untuk memilih fitur lainnya.")
    st.markdown("---")
    
    st.subheader("Informasi Dataset")
    st.markdown("""
    Dataset yang digunakan untuk melatih model ini adalah ulasan dari aplikasi Kopi Kenangan yang diambil dari Google Play Store.
    - **Sumber:** Google Play Store
    - **Bahasa:** Indonesia
    - **Jumlah Data Awal:** 25.000 ulasan
    - **Jumlah Data Setelah Preprocessing:** 16.942 ulasan
    """)
    st.markdown("---")

    st.markdown("**Distribusi Sentimen pada Keseluruhan Dataset (Setelah Data Cleaning, Normalisasi Teks dan Labelisasi)**")
    dataset_labelization_sentiment_dist = {'Positif': 6928, 'Negatif': 6359, 'Netral': 3655} 
    df_dataset_labelization_total_dist = pd.DataFrame(list(dataset_labelization_sentiment_dist.items()), columns=['Sentimen', 'Jumlah'])
    dataset_labelization_sentiment_dist_total = sum(dataset_labelization_sentiment_dist.values())

    fig_total, ax_total = plt.subplots(figsize=(10, 7))
    ax_total.bar(df_dataset_labelization_total_dist['Sentimen'].str.capitalize(), df_dataset_labelization_total_dist['Jumlah'], color=['#34A853', '#EA4335', '#FBBC05'])
    fig_total.patch.set_alpha(0.0)
    ax_total.patch.set_alpha(0.0)
    st.pyplot(fig_total)

    st.markdown(f"""
    - **Positif:** {dataset_labelization_sentiment_dist['Positif']} ulasan
    - **Negatif:** {dataset_labelization_sentiment_dist['Negatif']} ulasan
    - **Netral:** {dataset_labelization_sentiment_dist['Netral']} ulasan
    - **Total Ulasan:** {dataset_labelization_sentiment_dist_total} ulasan
    """)
    st.markdown("---")

    st.markdown("**Distribusi Sentimen pada Dataset Splitting dengan Rasio 70:15:15**")
    dataset_splitting_sentiment_dist = {'Training': 11859, 'Validation': 2541, 'Test': 2542}
    df_dataset_splitting_total_dist = pd.DataFrame(list(dataset_splitting_sentiment_dist.items()), columns=['Sentimen', 'Jumlah'])
    dataset_splitting_sentiment_dist_total = sum(dataset_splitting_sentiment_dist.values())

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(df_dataset_splitting_total_dist['Sentimen'], df_dataset_splitting_total_dist['Jumlah'], color=["#0040FF", "#00FF00", '#FFD700'])
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    st.pyplot(fig)

    st.markdown(f"""
    - **Training Set:** {dataset_splitting_sentiment_dist['Training']} ulasan
    - **Validation Set:** {dataset_splitting_sentiment_dist['Validation']} ulasan
    - **Test Set:** {dataset_splitting_sentiment_dist['Test']} ulasan
    - **Total Ulasan:** {dataset_splitting_sentiment_dist_total} ulasan
    """)
    st.markdown("---")

    st.subheader("Informasi Model")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        Model yang digunakan adalah **IndoBERT (indobenchmark/indobert-base-p2)** yang telah di-fine-tuning untuk tugas klasifikasi sentimen 3 kelas (Positif, Negatif, Netral).
        """)
    
    with col4:
        test_accuracy = 87.14 
        test_f1_score = 0.87

        st.metric("Akurasi", f"{test_accuracy}%")
        st.metric("F1-Score Rata-rata", f"{test_f1_score}")
        st.markdown("---")

# Konten Fitur Simulator Preprocessing Teks Ulasan
elif page == "Simulator Preprocessing Teks Ulasan":
    st.title("🔬 Simulator Preprocessing Teks Ulasan")
    st.write("Halaman ini mendemonstrasikan bagaimana teks ulasan mentah diolah dengan dengan teknik data cleaning dan normalisasi kata sebelum lanjut ke tahapan lainnya.")
    
    def update_text_area_simulator():
        st.session_state.simulator_input = st.session_state.text_area_simulator_key

    st.text_area("Masukkan teks kotor/noisy dengan singkatan, angka, emoji, atau tanda baca untuk melihat proses transformasinya. Contoh: 'Diupdate kok malah makin lemot?? kmrn lancar aj skrg force close trus, gila sih ni app 😡 ga rekomen' atau klik tombol 'Coba Sebuah Contoh Ulasan Secara Acak' dibawah :", key='text_area_simulator_key', value=st.session_state.simulator_input, on_change=update_text_area_simulator, height=150)

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        if st.button("Coba Sebuah Contoh Ulasan Secara Acak", use_container_width=True):
            if raw_samples:
                st.session_state.simulator_input = pd.Series(raw_samples).sample(1).iloc[0]
                st.rerun()
    with sim_col2:
        if st.button("Hapus Seluruh Ulasan", use_container_width=True):
            st.session_state.simulator_input = ""
            st.rerun()
    
    if st.button("Proses Teks", use_container_width=True):
        text_to_process = st.session_state.simulator_input
        if text_to_process:
            st.markdown("---")
            st.subheader("Hasil Transformasi Teks")
            st.markdown("**1. Teks Asli dari Pengguna**"); st.code(text_to_process, language='text')
            cleaned_text = clean_text(text_to_process)
            st.markdown("**2. Hasil Cleaning (Menghilangkan noise, emoji, tanda baca, dan angka)**"); st.code(cleaned_text, language='text')
            normalized_text = normalize_text(cleaned_text, normalization_dict)
            st.markdown("**3. Hasil Normalisasi (Mengubah singkatan menjadi kata baku)**"); st.code(normalized_text, language='text')
            st.success("Teks di atas adalah contoh teks yang akan digunakan untuk dilabelisasi, tokenisasi dan dianalisis oleh model.")
        else:
            st.warning("Mohon masukkan teks untuk diproses.")

# Konten Fitur Analisis Sentimen Tunggal
elif page == "Analisis Sentimen Tunggal":
    st.title("✍️ Analisis Sentimen Tunggal")
    st.header("Analisis Sebuah Ulasan")
    def update_text_area():
        st.session_state.user_input = st.session_state.text_area_key
        
    st.text_area("Masukkan ulasan Anda pada kolom text area dibawah ini atau klik tombol 'Coba Sebuah Contoh Ulasan Secara Acak' dibawah :", key='text_area_key', value=st.session_state.user_input, on_change=update_text_area, height=150)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Coba Sebuah Contoh Ulasan Secara Acak", use_container_width=True):
            if processed_samples:
                st.session_state.user_input = pd.Series(processed_samples).sample(1).iloc[0]
                st.rerun()
    
    with col2:
        if st.button("Hapus Seluruh Ulasan", use_container_width=True):
            st.session_state.user_input = ""
            st.rerun()
    
    if st.button("Analisis Ulasan Sekarang", use_container_width=True):
        text_to_analyze = st.session_state.user_input
        if text_to_analyze and model:
            with st.spinner('Menganalisis...'):
                prediction, all_probs = predict_sentiment(text_to_analyze, model, tokenizer, label_encoder, device)
            st.subheader("Hasil Analisis:")
            if prediction == "positif": st.success(f"Sentimen: **{prediction.capitalize()}**")
            elif prediction == "negatif": st.error(f"Sentimen: **{prediction.capitalize()}**")
            else: st.info(f"Sentimen: **{prediction.capitalize()}**")
            st.markdown("---")
            st.subheader("Confidence score (Probabilitas)")
            sorted_probs = sorted(all_probs.items(), key=lambda item: item[1], reverse=True)
            for label, prob in sorted_probs:
                st.write(f"**{label.capitalize()}**: {prob:.2%}")
        elif not model: st.error("Model tidak dapat dimuat. Aplikasi tidak bisa berjalan.")
        else: st.warning("Mohon masukkan sebuah ulasan untuk dianalisis.")

# Konten Fitur Analisis Sentimen Batch
elif page == "Analisis Sentimen Batch":
    st.title("🗂️ Analisis Sentimen Batch")
    st.header("Analisis Beberapa Ulasan Sekaligus")
    def update_text_area_batch():
        st.session_state.batch_input = st.session_state.text_area_batch_key

    st.text_area("Masukkan beberapa ulasan (satu ulasan per baris) pada kolom text area dibawah ini atau klik tombol 'Coba 3 Contoh Ulasan Secara Acak' dibawah :", key='text_area_batch_key', value=st.session_state.batch_input, on_change=update_text_area_batch, height=200)
    
    col_batch_1, col_batch_2 = st.columns(2)
    with col_batch_1:
        if st.button("Coba 3 Contoh Ulasan Secara Acak", use_container_width=True):
            if processed_samples and len(processed_samples) >= 3:
                random_samples = pd.Series(processed_samples).sample(3).tolist()
                st.session_state.batch_input = '\n'.join(random_samples)
                st.rerun()
            elif processed_samples:
                st.warning(f"Hanya ada {len(processed_samples)} contoh, tidak cukup untuk mengambil 3 Ulasan.")
    
    with col_batch_2:
        if st.button("Hapus Seluruh Ulasan", use_container_width=True):
            st.session_state.batch_input = ""
            st.rerun()
    
    if st.button("Analisis Semua Ulasan Sekarang", use_container_width=True):
        text_to_analyze_batch = st.session_state.batch_input
        if text_to_analyze_batch and model:
            list_ulasan_mentah = [ulasan.strip() for ulasan in text_to_analyze_batch.split('\n') if ulasan.strip()]
            if list_ulasan_mentah:
                hasil_prediksi = []
                list_ulasan_bersih = []
                progress_bar = st.progress(0, text="Menganalisis ulasan...")
                for i, ulasan in enumerate(list_ulasan_mentah):
                    ulasan_bersih = full_preprocess(ulasan)
                    list_ulasan_bersih.append(ulasan_bersih)
                    prediksi, _ = predict_sentiment(ulasan_bersih, model, tokenizer, label_encoder, device)
                    hasil_prediksi.append(prediksi)
                    progress_bar.progress((i + 1) / len(list_ulasan_mentah), text=f"Ulasan {i+1}/{len(list_ulasan_mentah)}")
                progress_bar.empty()
                
                jumlah_ulasan = len(list_ulasan_mentah)
                hitungan_sentimen = Counter(hasil_prediksi)
                st.subheader("Hasil Analisis Sentimen Ulasan Batch")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Ulasan", f"{jumlah_ulasan}")
                col2.metric("Positif", f"{hitungan_sentimen.get('positif', 0)}")
                col3.metric("Negatif", f"{hitungan_sentimen.get('negatif', 0)}")
                col4.metric("Netral", f"{hitungan_sentimen.get('netral', 0)}")

                visualisasi_1, visualisasi_2 = st.columns([1, 1])
                with visualisasi_1:
                    st.subheader("Distribusi Sentimen")
                    sentiment_order = ['positif', 'negatif', 'netral']
                    colors_map = {'positif': '#34A853', 'negatif': '#EA4335', 'netral': '#FBBC05'}
                    chart_labels, chart_sizes, chart_colors = [], [], []
                    for sentiment in sentiment_order:
                        if hitungan_sentimen[sentiment] > 0:
                            chart_labels.append(sentiment.capitalize())
                            chart_sizes.append(hitungan_sentimen[sentiment])
                            chart_colors.append(colors_map[sentiment])
                    if chart_sizes:
                        fig, ax = plt.subplots(figsize=(5, 5))
                        ax.pie(chart_sizes, labels=chart_labels, autopct='%1.1f%%', startangle=90, colors=chart_colors, textprops={'color':"w", 'weight':'bold'})
                        ax.legend(chart_labels, loc="best")
                        fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
                        st.pyplot(fig)
                with visualisasi_2:
                    st.subheader("Rincian Hasil Analisis")
                    df_hasil = pd.DataFrame({'Ulasan': list_ulasan_mentah, 'Prediksi Sentimen': [p.capitalize() for p in hasil_prediksi]})
                    st.dataframe(df_hasil, height=300, use_container_width=True)
                
                st.markdown("---")
                st.subheader("Word Cloud dari Kumpulan Ulasan")
                all_text = ' '.join(list_ulasan_bersih)
                if all_text:
                    wordcloud = WordCloud(width=1200, height=600, background_color='white', collocations=False).generate(all_text)
                    fig_wc, ax_wc = plt.subplots(figsize=(10,5))
                    ax_wc.imshow(wordcloud, interpolation='bilinear')
                    ax_wc.axis("off")
                    st.pyplot(fig_wc)
                else:
                    st.write("Tidak cukup kata untuk membuat Word Cloud.")

            else:
                st.warning("Mohon masukkan setidaknya satu ulasan.")
        elif not model:
            st.error("Model tidak dapat dimuat. Aplikasi tidak bisa berjalan.")
        else:
            st.warning("Anda belum memasukan ulasan satupun. Mohon masukan ulasan pada kotak input")