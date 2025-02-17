import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar el modelo
MODEL_PATH = "modelo_pisos1_2.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Configuración de la interfaz
st.title("Detector de Pisos en Casas 🏠")
st.write("Sube una o varias imágenes de casas y el modelo intentará predecir cuántos pisos tienen.")

# Subir imágenes
uploaded_files = st.file_uploader("Elige una o varias imágenes...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Imagen subida: {uploaded_file.name}", use_container_width=True)
        
        # Preprocesar la imagen
        img_resized = image.resize((128, 128))  # Redimensionar al tamaño esperado por el modelo
        img_array = np.array(img_resized) / 255.0  # Normalizar
        img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión batch
        
        # Hacer la predicción
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        
        # Mapear la clase a número de pisos (ajusta según tu dataset)
        clases = {0: "1 piso", 1: "2 pisos", 2: "3 pisos", 3: "4 pisos", 4: "5 pisos", 5: "6 pisos"}
        resultado = clases.get(predicted_class, "Desconocido")
        
        # Mostrar resultado
        st.success(f"El modelo predice que la casa tiene: {resultado}")
