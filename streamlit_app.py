#streamlit_app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np
from PIL import Image
import io
import os # Impor os untuk memeriksa path

# Atur konfigurasi halaman sebagai perintah Streamlit pertama
st.set_page_config(page_title="Prediktor Angka Digambar", layout="wide")

# Memuat model Keras yang telah dilatih
# Pastikan file 'Costum MobileNet.h5' berada di direktori yang sama
# atau sediakan path yang benar ke file model.
MODEL_PATH = 'CostumMobileNet.h5'

# Menggunakan cache untuk memuat model agar lebih efisien
@st.cache_resource
def load_keras_model():
    """Memuat model Keras dari path yang ditentukan."""
    if not os.path.exists(MODEL_PATH):
        return FileNotFoundError(f"File model '{MODEL_PATH}' tidak ditemukan. Pastikan file berada di direktori yang benar: {os.getcwd()}")
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        return e # Kembalikan exception untuk ditangani nanti

model_or_error = load_keras_model()

# Daftar nama kelas (angka 0-9)
class_names = [str(i) for i in range(10)]

st.title("✍️ Prediktor Angka Digambar Tangan")
st.markdown("""
Gambar satu digit angka (0-9) pada kanvas di bawah ini dan klik tombol "Prediksi" untuk melihat hasilnya.
Model yang digunakan adalah MobileNet Tranfer Learning.
""")

# Tambahkan pesan jika model tidak ditemukan di awal
if isinstance(model_or_error, FileNotFoundError):
    st.error(str(model_or_error))
    st.info(f"Pastikan file '{MODEL_PATH}' berada di direktori yang sama dengan skrip aplikasi Anda, atau perbarui variabel `MODEL_PATH` dalam kode dengan path yang benar.")
elif isinstance(model_or_error, Exception):
    st.error(f"Terjadi error lain saat memuat model: {model_or_error}")

# Kolom untuk kanvas dan output
col1, col2 = st.columns([1, 1])

# Atur nilai default untuk parameter kanvas di sini
STROKE_WIDTH = 20
STROKE_COLOR = "#000000" # Hitam
BG_COLOR = "#FFFFFF" # Putih
DRAWING_MODE = "freedraw"

with col1:
    st.subheader("Kanvas Gambar")
    # Buat kanvas untuk menggambar
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
        display_toolbar=False # Toolbar dihilangkan
    )
    st.caption("Gambar angka Anda di atas. Gunakan warna goresan hitam dan latar belakang putih untuk hasil terbaik.")


with col2:
    st.subheader("Hasil Prediksi")
    
    if isinstance(model_or_error, Exception):
        model = None
    else:
        model = model_or_error

    if model is None:
        st.warning("Model tidak berhasil dimuat atau error. Tidak dapat melakukan prediksi.")
    elif canvas_result.image_data is not None and canvas_result.image_data.shape[0] > 0 and canvas_result.image_data.shape[1] > 0:
        if st.button("Prediksi Angka", use_container_width=True, type="primary"):
            with st.spinner("Memproses gambar dan melakukan prediksi..."):
                img_data_rgba = canvas_result.image_data
                pil_image = Image.fromarray(img_data_rgba.astype('uint8'), 'RGBA')

                if BG_COLOR.lower() == '#ffffff00' or pil_image.mode == 'RGBA':
                    rgb_image = Image.new("RGB", pil_image.size, (255, 255, 255))
                    rgb_image.paste(pil_image, mask=pil_image.split()[3])
                else:
                    rgb_image = pil_image.convert('RGB')

                img_resized = rgb_image.resize((224, 224))
                img_array = img_to_array(img_resized)
                img_preprocessed = preprocess_input(np.expand_dims(img_array, axis=0))

                try:
                    prediction = model.predict(img_preprocessed)
                    predicted_class_index = np.argmax(prediction[0])
                    predicted_class_name = class_names[predicted_class_index]
                    confidence = np.max(prediction[0]) * 100

                    st.success(f"Angka yang Diprediksi: **{predicted_class_name}**")
                    st.info(f"Tingkat Kepercayaan: **{confidence:.2f}%**")
                    st.image(img_resized, caption="Gambar yang Diproses (224x224)", width=150)
                    
                    st.subheader("Probabilitas per Kelas:")
                    probs_dict = {class_names[i]: prediction[0][i]*100 for i in range(len(class_names))}
                    st.bar_chart(probs_dict)

                except Exception as e:
                    st.error(f"Terjadi error saat prediksi: {e}")
        else:
            st.info("Gambar sesuatu di kanvas dan klik tombol prediksi.")
    else:
        st.info("Kanvas kosong. Silakan gambar sebuah angka.")

if isinstance(model_or_error, FileNotFoundError):
    st.sidebar.error(str(model_or_error))
    st.sidebar.info(f"Pastikan file '{MODEL_PATH}' berada di direktori: {os.getcwd()}")
elif isinstance(model_or_error, Exception):
    st.sidebar.error(f"Error lain saat memuat model: {model_or_error}")
elif model is None and not isinstance(model_or_error, Exception):
    st.sidebar.warning(f"File model '{MODEL_PATH}' tidak dapat dimuat karena alasan yang tidak diketahui. Periksa log.")

# streamlit run "d:\PYTHON\Angka Prediction\streamlit_app.py"
# https://www.kaggle.com/datasets/olafkrastovski/handwritten-digits-0-9/data
# https://colab.research.google.com/drive/1MTproJzZmfAzRYDJhBwbuOKtn2Ia5wDP?authuser=0#scrollTo=YYWi06sIh7DP