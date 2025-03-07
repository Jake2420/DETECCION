import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from io import BytesIO

# Cargar modelo
model = tf.keras.models.load_model("Prueba.h5")
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])  # 🔹 Evita el warning de métricas vacías

st.title("Detector de Pisos en Casas 🏠")
st.write("Sube imágenes de casas y el modelo intentará predecir cuántos pisos tienen.")

if "resultados" not in st.session_state:
    st.session_state["resultados"] = []

uploaded_files = st.file_uploader("Elige imágenes...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Cargar imagen y mostrarla
        img = Image.open(uploaded_file)
        img = np.array(img)
        st.image(img, caption=f"Imagen subida: {uploaded_file.name}", use_container_width=True)

        # NO convertir a escala de grises si el modelo fue entrenado con RGB
        img = cv2.resize(img, (100, 100))  # 🔹 Mantener tamaño fijo
        img = img / 255.0  # 🔹 Normalizar
        img = np.expand_dims(img, axis=0)  # 🔹 Agregar batch dimension

        # Convertir a tensor para evitar warnings
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

        # Verificar dimensiones de entrada antes de predecir
        st.write(f"Dimensiones de la imagen procesada: {img_tensor.shape}")
        
        # Hacer predicción
        predictions = model.predict(img_tensor)
        predicted_class = np.argmax(predictions)  # Obtener la clase con mayor probabilidad
        
        # Mapear clases
        clases = {0: "1 piso", 1: "2 pisos", 2: "3 pisos", 3: "4 pisos", 4: "5 pisos", 5: "6 pisos"}
        resultado = clases.get(predicted_class, "Desconocido")

        # Se evita duplicados 
        if [uploaded_file.name, resultado] not in st.session_state["resultados"]:
            st.session_state["resultados"].append([uploaded_file.name, resultado])

        st.success(f"El modelo predice que la casa tiene: {resultado}")
        

def exportar_a_excel():
    if not st.session_state["resultados"]:
        st.warning("No hay resultados para exportar.")
        return None

    df = pd.DataFrame(st.session_state["resultados"], columns=["Imagen", "Predicción"])
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Resultados")
    output.seek(0)
    return output

st.header("Exportar Resultados")
if st.button("Guardar Excel"):
    excel_file = exportar_a_excel()
    if excel_file:
        st.download_button(
            label="📥 Descargar Excel",
            data=excel_file,
            file_name="resultados_pisos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

st.header("Historial de Resultados")
if st.session_state["resultados"]:
    df = pd.DataFrame(st.session_state["resultados"], columns=["Imagen", "Predicción"])
    st.dataframe(df)
else:
<<<<<<< HEAD
    st.write("No hay resultados almacenados aún.")
=======
    st.write("No hay resultados almacenados aún.")
>>>>>>> 908c8c3 (Corrección de errores en la predicción del modelo)
