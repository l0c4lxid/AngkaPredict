# streamlit_app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np
from PIL import Image
import os
from tempfile import NamedTemporaryFile

# Konfigurasi halaman
st.set_page_config(page_title="Prediktor Angka Digambar", layout="wide")

# Sidebar untuk upload model
st.sidebar.header("⚙️ Konfigurasi Model")
uploaded_model_file = st.sidebar.file_uploader("📁 Upload file model (.h5)", type=["h5"])

model = None
model_error = None

if uploaded_model_file is not None:
    try:
        with NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(uploaded_model_file.read())
            tmp_path = tmp.name
        model = load_model(tmp_path)
        st.sidebar.success("✅ Model berhasil dimuat!")
    except Exception as e:
        model_error = e
        st.sidebar.error(f"❌ Gagal memuat model: {e}")
else:
    st.sidebar.info("Silakan upload file model (.h5) untuk mulai.")

# Daftar nama kelas (0-9)
class_names = [str(i) for i in range(10)]

# Judul dan instruksi
st.title("✍️ Prediktor Angka Digambar Tangan")
st.markdown("""
Gambar satu digit angka (0-9) pada kanvas di bawah ini dan klik tombol "Prediksi" untuk melihat hasilnya.
Model yang digunakan adalah MobileNet Transfer Learning.
""")

# Layout 2 kolom
col1, col2 = st.columns([1, 1])

# Parameter kanvas
STROKE_WIDTH = 20
STROKE_COLOR = "#000000"
BG_COLOR = "#FFFFFF"
DRAWING_MODE = "freedraw"

# Kolom kiri: Kanvas
with col1:
    st.subheader("Kanvas Gambar")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=STROKE_WIDTH,
        stroke_color=STROKE_COLOR,
        background_color=BG_COLOR,
        update_streamlit=True,
        height=300,
        width=300,
        drawing_mode=DRAWING_MODE,
        key="digit_canvas",
        display_toolbar=False
    )
    st.caption("Gambar angka Anda di atas. Gunakan warna hitam dan latar belakang putih untuk hasil terbaik.")

# Kolom kanan: Prediksi
with col2:
    st.subheader("Hasil Prediksi")

    if model_error is not None:
        st.warning(f"Model gagal dimuat: {model_error}")
    elif model is None:
        st.info("Silakan upload model terlebih dahulu untuk mulai.")
    elif canvas_result.image_data is not None and canvas_result.image_data.shape[0] > 0:
        if st.button("Prediksi Angka", use_container_width=True, type="primary"):
            with st.spinner("Memproses gambar dan melakukan prediksi..."):
                try:
                    # Gambar dari canvas -> RGBA -> RGB
                    img_data_rgba = canvas_result.image_data
                    pil_image = Image.fromarray(img_data_rgba.astype('uint8'), 'RGBA')
                    rgb_image = Image.new("RGB", pil_image.size, (255, 255, 255))
                    rgb_image.paste(pil_image, mask=pil_image.split()[3])

                    # Resize dan preprocess
                    img_resized = rgb_image.resize((224, 224))
                    img_array = img_to_array(img_resized)
                    img_preprocessed = preprocess_input(np.expand_dims(img_array, axis=0))

                    # Prediksi
                    prediction = model.predict(img_preprocessed)
                    predicted_index = np.argmax(prediction[0])
                    predicted_label = class_names[predicted_index]
                    confidence = np.max(prediction[0]) * 100

                    # Tampilkan hasil
                    st.success(f"Angka yang Diprediksi: **{predicted_label}**")
                    st.info(f"Tingkat Kepercayaan: **{confidence:.2f}%**")
                    st.image(img_resized, caption="Gambar yang Diproses (224x224)", width=150)

                    # Tampilkan probabilitas semua kelas
                    st.subheader("Probabilitas per Kelas:")
                    probs_dict = {class_names[i]: prediction[0][i] * 100 for i in range(len(class_names))}
                    st.bar_chart(probs_dict)

                except Exception as e:
                    st.error(f"❌ Terjadi error saat prediksi: {e}")
        else:
            st.info("Gambar sesuatu di kanvas lalu klik tombol prediksi.")
    else:
        st.info("Kanvas kosong. Silakan gambar angka terlebih dahulu.")
