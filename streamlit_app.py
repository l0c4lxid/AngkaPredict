import streamlit as st
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np
from PIL import Image
import os

MODEL_DIR = "models"
MODEL_FILENAME = "CostumMobileNet.h5"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Pastikan folder model ada
os.makedirs(MODEL_DIR, exist_ok=True)

st.set_page_config(page_title="Prediktor Angka Digambar", layout="wide")

st.sidebar.header("⚙️ Upload Model (Admin)")

# Cek apakah model sudah ada di server
model_exists = os.path.exists(MODEL_PATH)

uploaded_file = None
model = None
model_error = None

# Kalau belum ada, minta upload
if not model_exists:
    uploaded_file = st.sidebar.file_uploader("Upload file model (.h5)", type=["h5"])
    if uploaded_file is not None:
        # Simpan file yang diupload ke folder models
        with open(MODEL_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Model tersimpan di server: {MODEL_PATH}")
        model_exists = True

# Kalau model sudah ada di server, langsung load
if model_exists:
    try:
        model = load_model(MODEL_PATH)
        st.sidebar.success(f"Model dimuat dari {MODEL_PATH}")
    except Exception as e:
        model_error = e
        st.sidebar.error(f"Gagal memuat model: {e}")

class_names = [str(i) for i in range(10)]

st.title("✍️ Prediktor Angka Digambar Tangan")
st.markdown("""
Gambar satu digit angka (0-9) pada kanvas di bawah ini dan klik tombol "Prediksi" untuk melihat hasilnya.
Model yang digunakan adalah MobileNet Transfer Learning.
""")

col1, col2 = st.columns([1,1])

STROKE_WIDTH = 20
STROKE_COLOR = "#000000"
BG_COLOR = "#FFFFFF"
DRAWING_MODE = "freedraw"

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

with col2:
    st.subheader("Hasil Prediksi")

    if model_error is not None:
        st.warning(f"Model gagal dimuat: {model_error}")
    elif model is None:
        st.info("Model belum tersedia. Silakan upload model dulu di sidebar.")
    elif canvas_result.image_data is not None and canvas_result.image_data.shape[0] > 0:
        if st.button("Prediksi Angka", use_container_width=True, type="primary"):
            with st.spinner("Memproses gambar dan melakukan prediksi..."):
                try:
                    img_data_rgba = canvas_result.image_data
                    pil_image = Image.fromarray(img_data_rgba.astype('uint8'), 'RGBA')
                    rgb_image = Image.new("RGB", pil_image.size, (255, 255, 255))
                    rgb_image.paste(pil_image, mask=pil_image.split()[3])

                    img_resized = rgb_image.resize((224, 224))
                    img_array = img_to_array(img_resized)
                    img_preprocessed = preprocess_input(np.expand_dims(img_array, axis=0))

                    prediction = model.predict(img_preprocessed)
                    predicted_index = np.argmax(prediction[0])
                    predicted_label = class_names[predicted_index]
                    confidence = np.max(prediction[0]) * 100

                    st.success(f"Angka yang Diprediksi: **{predicted_label}**")
                    st.info(f"Tingkat Kepercayaan: **{confidence:.2f}%**")
                    st.image(img_resized, caption="Gambar yang Diproses (224x224)", width=150)

                    st.subheader("Probabilitas per Kelas:")
                    probs_dict = {class_names[i]: prediction[0][i] * 100 for i in range(len(class_names))}
                    st.bar_chart(probs_dict)

                except Exception as e:
                    st.error(f"Terjadi error saat prediksi: {e}")
        else:
            st.info("Gambar sesuatu di kanvas dan klik tombol prediksi.")
    else:
        st.info("Kanvas kosong. Silakan gambar angka.")

