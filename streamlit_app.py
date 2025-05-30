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

# Konfigurasi halaman
st.set_page_config(page_title="Prediktor Angka Digambar", layout="wide")

# Path model
MODEL_PATH = 'CostumMobileNet.h5'

# Load model dengan cache
@st.cache_resource
def load_keras_model():
    if not os.path.exists(MODEL_PATH):
        return FileNotFoundError(f"File model '{MODEL_PATH}' tidak ditemukan di: {os.getcwd()}")
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        return e

model_or_error = load_keras_model()
class_names = [str(i) for i in range(10)]

st.title("✍️ Prediktor Angka Digambar Tangan")
st.markdown("Gambar satu digit angka (0-9) lalu klik **Prediksi**. Model menggunakan MobileNet Transfer Learning.")

# Cek apakah model valid
if isinstance(model_or_error, FileNotFoundError):
    st.error(str(model_or_error))
    st.stop()
elif isinstance(model_or_error, Exception):
    st.error(f"Terjadi error saat memuat model: {model_or_error}")
    st.stop()
else:
    model = model_or_error

# Kolom layout
col1, col2 = st.columns([1, 1])

# === BAGIAN KIRI: KANVAS ===
with col1:
    st.subheader("🖌️ Kanvas Gambar")

    # Session state untuk reset canvas
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = "canvas"

    # Tombol Reset
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

    st.caption("Gambar angka di atas (gunakan warna hitam dan latar putih).")

# === BAGIAN KANAN: PREDIKSI ===
with col2:
    st.subheader("📊 Hasil Prediksi")

    if canvas_result.image_data is not None and canvas_result.image_data.shape[0] > 0:
        if st.button("✅ Prediksi Angka", type="primary"):
            with st.spinner("Memproses gambar..."):
                # Ambil gambar RGBA dari canvas
                img_data_rgba = canvas_result.image_data
                pil_image = Image.fromarray(img_data_rgba.astype('uint8'), 'RGBA')

                # Konversi ke RGB
                rgb_image = Image.new("RGB", pil_image.size, (255, 255, 255))
                rgb_image.paste(pil_image, mask=pil_image.split()[3])

                # Resize dan preprocess
                img_resized = rgb_image.resize((224, 224))
                img_array = img_to_array(img_resized)
                img_preprocessed = preprocess_input(np.expand_dims(img_array, axis=0))

                try:
                    prediction = model.predict(img_preprocessed)
                    predicted_class_index = np.argmax(prediction[0])
                    predicted_class_name = class_names[predicted_class_index]
                    confidence = np.max(prediction[0]) * 100

                    # Hasil
                    st.success(f"🎯 Prediksi: **{predicted_class_name}**")
                    st.info(f"Tingkat Kepercayaan: **{confidence:.2f}%**")
                    st.image(img_resized, caption="Gambar yang Diproses (224x224)", width=150)

                    # Chart
                    st.subheader("Distribusi Probabilitas:")
                    st.bar_chart({class_names[i]: float(prediction[0][i]*100) for i in range(10)})

                except Exception as e:
                    st.error(f"Gagal prediksi: {e}")
        else:
            st.info("Klik **Prediksi Angka** setelah menggambar.")
    else:
        st.info("Kanvas masih kosong. Silakan gambar dulu.")

# Jalankan dengan perintah:
# streamlit run streamlit_app.py
