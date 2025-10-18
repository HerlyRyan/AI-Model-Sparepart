import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ===============================
# 🌐 UI STREAMLIT
# ===============================
st.set_page_config(page_title="AI Image Classifier", page_icon="🧠", layout="centered")

# ===============================
# 🧠 CONFIGURASI DASAR
# ===============================
MODEL_PATH = "model/model_trained.tflite"   # atau "model/model_trained.tflite"
LABEL_PATH = "model/labels.txt"
TARGET_SIZE = (224, 224)

# ===============================
# 🚀 LOAD LABEL SECARA OTOMATIS
# ===============================
if os.path.exists(LABEL_PATH):
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        CLASS_NAMES = [line.strip() for line in f.readlines() if line.strip()]
else:
    CLASS_NAMES = []
    st.warning("⚠️ File labels.txt tidak ditemukan. Pastikan sudah ada di folder model/")

# ===============================
# 🧩 LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    if MODEL_PATH.endswith(".tflite"):
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter, "tflite"
    else:
        # Load H5 (Keras)
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, "keras"

model, model_type = load_model()
st.success(f"✅ Model berhasil dimuat ({model_type.upper()})")

# ===============================
# 🧮 FUNGSI PREDIKSI
# ===============================
def predict_image(image: Image.Image):
    img = image.convert("RGB").resize(TARGET_SIZE)
    input_data = np.expand_dims(np.array(img) / 255.0, axis=0)

    if model_type == "keras":
        preds = model.predict(input_data)[0]
    else:
        input_index = model.get_input_details()[0]["index"]
        output_index = model.get_output_details()[0]["index"]
        model.set_tensor(input_index, input_data.astype(np.float32))
        model.invoke()
        preds = model.get_tensor(output_index)[0]

    pred_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    label = CLASS_NAMES[pred_idx] if CLASS_NAMES else f"Class {pred_idx}"
    return label, confidence, preds

st.title("🧠 AI Image Classification Demo")
st.markdown("""
Upload sebuah gambar untuk diuji menggunakan model AI kamu.  
Model akan menampilkan **prediksi label** dan **tingkat kepercayaannya**.
""")

uploaded_file = st.file_uploader("📤 Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar diupload", use_column_width=True)

    st.write("🔍 Sedang menganalisis...")
    label, confidence, preds = predict_image(image)

    st.subheader("🎯 Hasil Prediksi:")
    st.markdown(f"**Label:** `{label}`")
    st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")

    # Tampilkan bar chart confidence semua kelas
    if CLASS_NAMES:
        st.bar_chart(
            data={CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
        )
else:
    st.info("⬆️ Silakan upload gambar untuk memulai prediksi.")

# ===============================
# 📊 Footer
# ===============================
st.markdown("---")
st.caption("Model AI Classification | Built with ❤️ using Streamlit + TensorFlow")
