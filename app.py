import streamlit as st
# import pickle
import pandas as pd
import numpy as np

# Cargar el modelo guardado
@st.cache_data
def cargar_modelo():
    with open('churn-model.pck', 'rb') as file:
        return pickle.load(file)

# Cargar el modelo
modelo_regresion = cargar_modelo()

# Título de la app
st.title("Predicción de Telco - Modelo de Regresión Lineal")

# Explicación de la app
st.write("""
Esta aplicación utiliza un modelo de regresión lineal entrenado sobre el dataset Telco para predecir características relacionadas con clientes de telecomunicaciones.
Introduce los valores de las variables para hacer una predicción.
""")

# Entradas del usuario
st.sidebar.header("Introduce las características del cliente")

# Variables categóricas y numéricas
gender = st.sidebar.selectbox("Género", ['Femenino', 'Masculino'])
seniorcitizen = st.sidebar.selectbox("Senior Citizen (1=Sí, 0=No)", [1, 0])
partner = st.sidebar.selectbox("Tiene pareja", ['Sí', 'No'])
dependents = st.sidebar.selectbox("Tiene dependientes", ['Sí', 'No'])
tenure = st.sidebar.number_input("Años de permanencia", min_value=0, max_value=72, value=12)
phoneservice = st.sidebar.selectbox("Tiene servicio telefónico", ['Sí', 'No'])
multiplelines = st.sidebar.selectbox("Tiene múltiples líneas", ['Sí', 'No', 'No phone service'])
internetservice = st.sidebar.selectbox("Servicio de Internet", ['Fibra óptica', 'DSL', 'No internet service'])
onlinesecurity = st.sidebar.selectbox("Seguridad online", ['Sí', 'No', 'No internet service'])
onlinebackup = st.sidebar.selectbox("Copia de seguridad online", ['Sí', 'No', 'No internet service'])
deviceprotection = st.sidebar.selectbox("Protección de dispositivo", ['Sí', 'No', 'No internet service'])
techsupport = st.sidebar.selectbox("Soporte técnico", ['Sí', 'No', 'No internet service'])
streamingtv = st.sidebar.selectbox("Streaming TV", ['Sí', 'No', 'No internet service'])
streamingmovies = st.sidebar.selectbox("Streaming Movies", ['Sí', 'No', 'No internet service'])
contract = st.sidebar.selectbox("Tipo de contrato", ['Mes a mes', 'Un año', 'Dos años'])
paperlessbilling = st.sidebar.selectbox("Facturación sin papel", ['Sí', 'No'])
paymentmethod = st.sidebar.selectbox("Método de pago", ['Banco', 'Cheque electrónico', 'Transferencia bancaria', 'Crédito automático'])
monthlycharges = st.sidebar.number_input("Cargo mensual", min_value=0, value=70)
totalcharges = st.sidebar.number_input("Cargo total", min_value=0, value=200)

# Preprocesar las variables categóricas
def preprocesar_datos(input_data):
    input_data['gender'] = input_data['gender'].apply(lambda x: 1 if x == 'Masculino' else 0)
    input_data['partner'] = input_data['partner'].apply(lambda x: 1 if x == 'Sí' else 0)
    input_data['dependents'] = input_data['dependents'].apply(lambda x: 1 if x == 'Sí' else 0)
    input_data['phoneservice'] = input_data['phoneservice'].apply(lambda x: 1 if x == 'Sí' else 0)
    input_data['multiplelines'] = input_data['multiplelines'].apply(
        lambda x: 1 if x == 'Sí' else (0 if x == 'No phone service' else -1)
    )
    input_data['internetservice'] = input_data['internetservice'].apply(
        lambda x: 0 if x == 'No internet service' else (1 if x == 'DSL' else 2)
    )
    input_data['onlinesecurity'] = input_data['onlinesecurity'].apply(
        lambda x: 1 if x == 'Sí' else (0 if x == 'No internet service' else -1)
    )
    input_data['onlinebackup'] = input_data['onlinebackup'].apply(
        lambda x: 1 if x == 'Sí' else (0 if x == 'No internet service' else -1)
    )
    input_data['deviceprotection'] = input_data['deviceprotection'].apply(
        lambda x: 1 if x == 'Sí' else (0 if x == 'No internet service' else -1)
    )
    input_data['techsupport'] = input_data['techsupport'].apply(
        lambda x: 1 if x == 'Sí' else (0 if x == 'No internet service' else -1)
    )
    input_data['streamingtv'] = input_data['streamingtv'].apply(
        lambda x: 1 if x == 'Sí' else (0 if x == 'No internet service' else -1)
    )
    input_data['streamingmovies'] = input_data['streamingmovies'].apply(
        lambda x: 1 if x == 'Sí' else (0 if x == 'No internet service' else -1)
    )
    input_data['contract'] = input_data['contract'].apply(
        lambda x: 0 if x == 'Mes a mes' else (1 if x == 'Un año' else 2)
    )
    input_data['paperlessbilling'] = input_data['paperlessbilling'].apply(
        lambda x: 1 if x == 'Sí' else 0
    )
    input_data['paymentmethod'] = input_data['paymentmethod'].apply(
        lambda x: 0 if x == 'Banco' else (1 if x == 'Cheque electrónico' else (2 if x == 'Transferencia bancaria' else 3))
    )
    return input_data


# Crear un DataFrame con los datos introducidos
nuevos_datos = pd.DataFrame({
    'gender': [gender],
    'seniorcitizen': [seniorcitizen],
    'partner': [partner],
    'dependents': [dependents],
    'tenure': [tenure],
    'phoneservice': [phoneservice],
    'multiplelines': [multiplelines],
    'internetservice': [internetservice],
    'onlinesecurity': [onlinesecurity],
    'onlinebackup': [onlinebackup],
    'deviceprotection': [deviceprotection],
    'techsupport': [techsupport],
    'streamingtv': [streamingtv],
    'streamingmovies': [streamingmovies],
    'contract': [contract],
    'paperlessbilling': [paperlessbilling],
    'paymentmethod': [paymentmethod],
    'monthlycharges': [monthlycharges],
    'totalcharges': [totalcharges]
})

# Preprocesar los datos antes de hacer la predicción
nuevos_datos_procesados = preprocesar_datos(nuevos_datos)

# Realizar la predicción con el modelo cargado
if st.sidebar.button('Predecir'):
    prediccion = modelo_regresion.predict(nuevos_datos_procesados)
    
    # Mostrar el resultado
    st.write(f"La predicción del modelo es: {prediccion[0]:.2f}")
