import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# ✅ Load the trained model
model = load_model("plant_model.h5")

# ✅ Define class labels (must match your dataset)
class_labels = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# ✅ Disease info
disease_info = {
    "Pepper__bell___Bacterial_spot": {
        "cause": "Bacterium (Xanthomonas campestris pv. vesicatoria)",
        "prevention": "Use disease-free seeds and avoid overhead watering."
    },
    "Pepper__bell___healthy": {
        "cause": "No disease",
        "prevention": "Maintain proper irrigation and nutrient levels."
    },
    "Potato___Early_blight": {
        "cause": "Fungus (Alternaria solani)",
        "prevention": "Use crop rotation and avoid excess moisture on leaves."
    },
    "Potato___Late_blight": {
        "cause": "Fungus (Phytophthora infestans)",
        "prevention": "Use resistant varieties and fungicide sprays."
    },
    "Potato___healthy": {
        "cause": "No disease",
        "prevention": "Keep soil well-drained and avoid waterlogging."
    },
    "Tomato_Bacterial_spot": {
        "cause": "Bacterium (Xanthomonas campestris pv. vesicatoria)",
        "prevention": "Avoid working with wet plants and use copper-based sprays."
    },
    "Tomato_Early_blight": {
        "cause": "Fungus (Alternaria solani)",
        "prevention": "Remove infected leaves and rotate crops yearly."
    },
    "Tomato_Late_blight": {
        "cause": "Fungus (Phytophthora infestans)",
        "prevention": "Avoid overhead watering and increase air circulation."
    },
    "Tomato_Leaf_Mold": {
        "cause": "Fungus (Passalora fulva)",
        "prevention": "Ensure good airflow and avoid high humidity."
    },
    "Tomato_Septoria_leaf_spot": {
        "cause": "Fungus (Septoria lycopersici)",
        "prevention": "Remove infected foliage and apply fungicide if needed."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "cause": "Pest (Tetranychus urticae)",
        "prevention": "Spray water on undersides of leaves and use miticides."
    },
    "Tomato__Target_Spot": {
        "cause": "Fungus (Corynespora cassiicola)",
        "prevention": "Avoid leaf wetness and use protective fungicides."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "cause": "Virus (Transmitted by whiteflies)",
        "prevention": "Control whiteflies and remove infected plants."
    },
    "Tomato__Tomato_mosaic_virus": {
        "cause": "Virus (Tobamovirus group)",
        "prevention": "Avoid tobacco use near plants and disinfect tools."
    },
    "Tomato_healthy": {
        "cause": "No disease",
        "prevention": "Maintain good nutrition and watering practices."
    }
}

# ✅ Streamlit UI
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")
st.title("🌿 Plant Disease Detection System")
st.write("Upload an image of a plant leaf to detect the disease, along with its cause and prevention methods.")

# File uploader
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    class_index = np.argmax(pred)
    class_label = class_labels[class_index]
    confidence = round(100 * np.max(pred), 2)

    # Display result
    st.subheader(f"🌿 Prediction: {class_label}")
    st.write(f"✅ Confidence: {confidence}%")

    if class_label in disease_info:
        cause = disease_info[class_label]['cause']
        prevention = disease_info[class_label]['prevention']
        st.write(f"🩺 **Cause:** {cause}")
        st.write(f"🌱 **Prevention:** {prevention}")
    else:
        cause = "Unknown"
        prevention = "No information available."
        st.write("⚠️ No detailed information available for this disease yet.")

    # ✅ Prepare downloadable report
    report_content = f"""
🌿 Plant Disease Detection Report
---------------------------------
Prediction: {class_label}
Confidence: {confidence}%

🩺 Cause:
{cause}

🌱 Prevention:
{prevention}

Thank you for using the Plant Disease Detection System!
"""

    # Convert report to bytes
    report_bytes = io.BytesIO(report_content.encode('utf-8'))

    # Download button
    st.download_button(
        label="📄 Download Report",
        data=report_bytes,
        file_name="Plant_Disease_Report.txt",
        mime="text/plain"
    )

st.markdown("---")
st.caption("Developed using TensorFlow & Streamlit 🌱")
