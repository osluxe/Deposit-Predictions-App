import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
with open('model-random-forest.pkl', 'rb') as file:
    model = joblib.load(file)

# Inisialisasi session state untuk menyimpan riwayat prediksi
if 'history' not in st.session_state:
    st.session_state.history = []

# Judul aplikasi
st.title("🎯 Prediksi Nasabah Berlangganan Deposito")
st.markdown("Masukkan data pada kolom berikut untuk mengetahui apakah nasabah diprediksi **berlangganan deposito atau tidak.**")

# Buat dua kolom input berdampingan
col1, col2 = st.columns(2)

with col1:
    duration = st.number_input(
        "Lama panggilan terakhir ke nasabah (detik)", 
        min_value=0, 
        value=319,  # rata-rata dari dataset
        help="Durasi telepon terakhir ke nasabah dalam detik. Rata-rata sekitar 319 detik."
    )

    euribor3m = st.number_input(
        "Suku bunga pasar Eropa (%)", 
        min_value=0.0,
        value=4.96,  # rata-rata dari dataset
        help="Suku bunga Euribor 3 bulan. Rentang umum: 0.5 - 5.0"
    )

    pdays = st.number_input(
        "jumlah hari dihubungi sebelum kampanye", 
        min_value=0,
        value=2,  # median mendekati 999 (belum pernah dihubungi)
        help="Jumlah hari sejak terakhir kontak. 999 berarti belum pernah dihubungi."
    )

    cons_price_idx = st.number_input(
        "Indikator inflasi", 
        min_value=0.0,
        value=85.00,  # rata-rata dari dataset
        help="Indeks harga konsumen. Rata-rata sekitar 93.58 (skala Eropa)."
    )

with col2:
    emp_var_rate = st.number_input(
        "Perubahan tingkat pekerjaan (%)", 
        format="%.2f",
        value=-1.01,  # rata-rata dari dataset
        help="Tingkat perubahan lapangan kerja. Rentang umum: -3.0 hingga +1.5"
    )

    campaign = st.number_input(
        "Jumlah panggilan selama kampanye", 
        min_value=0,
        value=2,
        help="Berapa kali nasabah dihubungi selama kampanye saat ini. Umumnya 1–3 kali."
    )

    cons_conf_idx = st.number_input(
        "Tingkat kepercayaan konsumen", 
        format="%.2f",
        value=-30.0,  # rata-rata dari dataset
        help="Indeks kepercayaan konsumen terhadap ekonomi. Umumnya antara -50 sampai -25."
    )

    nr_employed = st.number_input(
        "Jumlah tenaga kerja nasional (ribuan)", 
        min_value=0.0,
        value=5191.0,  # rata-rata dari dataset
        help="Jumlah karyawan secara nasional. Umumnya di kisaran 5000–5220."
    )


# Tombol prediksi
if st.button("🔍 Prediksi Sekarang"):
    input_data = np.array([[duration, euribor3m, pdays, cons_price_idx,
                            emp_var_rate, campaign, cons_conf_idx, nr_employed]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.success("✅ Nasabah diprediksi **DEPOSITO**.")
    else:
        st.warning("❌ Nasabah diprediksi **TIDAK DEPOSITO**.")

    # Simpan data input dan hasil ke riwayat (dengan label deskriptif)
    nasabah_data = {
        "Durasi Kontak (detik)": duration,
        "Euribor 3 Bulan": euribor3m,
        "Hari Sejak Terakhir Dihubungi": pdays,
        "Indeks Harga Konsumen": cons_price_idx,
        "Tingkat Variasi Kerja": emp_var_rate,
        "Jumlah Kontak Kampanye": campaign,
        "Indeks Kepercayaan Konsumen": cons_conf_idx,
        "Jumlah Karyawan (nr.employed)": nr_employed,
        "Prediksi": "Akan Deposito" if prediction[0] == 1 else "Tidak Deposito",
        "Probabilitas": round(probability, 2)
    }
    st.session_state.history.append(nasabah_data)

# Tampilkan riwayat prediksi
if st.session_state.history:
    st.subheader("📚 Riwayat Prediksi Sebelumnya")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)

    # Visualisasi Probabilitas
    st.subheader("📈 Visualisasi Probabilitas Deposito")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=history_df.index,
        y=history_df['Probabilitas'],
        hue=history_df['Prediksi'],
        palette='Set2',
        ax=ax
    )
    ax.set_title("Probabilitas Prediksi Deposito per Nasabah")
    ax.set_xlabel("Urutan Input")
    ax.set_ylabel("Probabilitas Kelas 1 (Deposito)")
    ax.set_ylim(0, 1)
    st.pyplot(fig)
