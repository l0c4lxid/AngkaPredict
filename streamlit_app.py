# streamlit_app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np
from PIL import Image
import os

# Konfigurasi
st.set_page_config(page_title="Prediktor Angka Digambar", layout="wide")
MODEL_PATH = "CostumMobileNet.h5"
class_names = [str(i) for i in range(10)]  # 0–9

# Fungsi: Load model jika sudah ada
@st.cache_resource
def load_keras_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        return e

# 1. CEK JIKA MODEL TIDAK ADA, TAMPILKAN TOMBOL UPLOAD
if not os.path.exists(MODEL_PATH):
    st.warning("⚠️ File model belum ada. Silakan upload file `.h5` terlebih dahulu.")
    uploaded_model = st.file_uploader("📤 Upload Model .h5", type=["h5"])

    if uploaded_model is not None:
        with open(MODEL_PATH, "wb") as f:
            f.write(uploaded_model.read())
        st.success("✅ Model berhasil diupload! Silakan refresh halaman.")
        st.stop()
    else:
        st.stop()

# 2. JIKA MODEL ADA, MAKA LANJUTKAN
model = load_keras_model()
if isinstance(model, Exception):
    st.error(f"❌ Gagal load model: {model}")
    st.stop()

# === UI ===
st.title("✍️ Prediktor Angka Digambar Tangan")
st.markdown("Gambar satu digit angka (0-9) lalu klik **Prediksi**. Model menggunakan MobileNet Transfer Learning.")

col1, col2 = st.columns(2)

# === BAGIAN KIRI: KANVAS ===
with col1:
    st.subheader("🖌️ Kanvas Gambar")

    # Inisialisasi kunci canvas
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = "canvas"

    if st.button("🔄 Reset Gambar"):
        st.session_state.canvas_key += "_new"

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=20,
        stroke_color="#000000",
        background_color="#FFFFFF",
        update_streamlit=True,
        height=300,
        width=300,
        drawing_mode="freedraw",
        key=st.session_state.canvas_key,
        display_toolbar=False
    )
    st.caption("Gambar angka (0-9) dengan warna hitam dan latar putih.")

# === BAGIAN KANAN: PREDIKSI ===
with col2:
    st.subheader("📊 Hasil Prediksi")

    if canvas_result.image_data is not None and canvas_result.image_data.shape[0] > 0:
        if st.button("✅ Prediksi Angka", type="primary"):
            with st.spinner("Memproses gambar..."):
                try:
                    # Ambil RGBA → RGB
                    img_data_rgba = canvas_result.image_data
                    pil_image = Image.fromarray(img_data_rgba.astype('uint8'), 'RGBA')
                    rgb_image = Image.new("RGB", pil_image.size, (255, 255, 255))
                    rgb_image.paste(pil_image, mask=pil_image.split()[3])

                    # Resize & preprocess
                    img_resized = rgb_image.resize((224, 224))
                    img_array = img_to_array(img_resized)
                    img_preprocessed = preprocess_input(np.expand_dims(img_array, axis=0))

                    prediction = model.predict(img_preprocessed)
                    pred_index = np.argmax(prediction[0])
                    pred_label = class_names[pred_index]
                    confidence = prediction[0][pred_index] * 100

                    st.success(f"🎯 Prediksi: **{pred_label}**")
                    st.info(f"Tingkat Kepercayaan: **{confidence:.2f}%**")
                    st.image(img_resized, caption="Gambar yang Diproses", width=150)

                    st.subheader("Distribusi Probabilitas:")
                    st.bar_chart({class_names[i]: float(prediction[0][i]*100) for i in range(10)})

                except Exception as e:
                    st.error(f"❌ Gagal memproses: {e}")
        else:
            st.info("Klik **Prediksi Angka** untuk memulai.")
    else:
        st.info("Kanvas masih kosong.")
