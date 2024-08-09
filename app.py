import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
import io

# Cargar datos del archivo Excel
df_empleados = pd.read_excel('empleados_ficticios_100.xlsx')

# Título de la aplicación
st.title("Gestión Inteligente de Capacitación y Promociones de Empleados")

# Sidebar - Sección de Ayuda
st.sidebar.header("Ayuda")
st.sidebar.write("""
**Integración de Bases de Datos Ficticios:**
- Visualiza los datos de empleados ficticios cargados desde un archivo Excel.
- Filtra empleados por subdirección y puesto actual para ver información específica.

**Simulación Avanzada:**
- **Distribución del Progreso en Capacitación:** Muestra la distribución del progreso en diferentes áreas de capacitación.
- **Simulación de Necesidades de Capacitación:** Calcula y visualiza las necesidades de capacitación basadas en el progreso actual.
- **Simulación de Impacto de Capacitación:** Muestra cómo un incremento en el progreso de capacitación afecta las necesidades de capacitación.

**Análisis de Correlación:**
- Muestra la matriz de correlación entre las diferentes áreas de progreso en capacitación.

**Automatización:**
- **Recomendación Automática de Roles y Cursos:** Recomienda cursos basados en el progreso de los empleados.
- **Recomendación de Promociones:** Sugiere posibles promociones basadas en el puesto actual.
- **Predicción de Ascensos:** Utiliza un modelo de Random Forest para predecir la probabilidad de ascenso de los empleados.

**Generar Reporte Personalizado:**
- Descarga un reporte en formato Excel con la información actualizada de los empleados.
""")

# Sidebar - Descripción sobre la IA y Pensionissste
st.sidebar.header("IA en Pensionissste")
st.sidebar.write("""
**Beneficios de la IA para Pensionissste:**
La inteligencia artificial (IA) puede ofrecer varias ventajas a Pensionissste, entre ellas:

1. **Análisis Predictivo:**
   - **Pronóstico de Necesidades de Capacitación:** La IA puede predecir qué empleados necesitarán capacitación adicional basándose en su progreso y desempeño.
   - **Predicción de Ascensos:** Modelos predictivos pueden ayudar a identificar empleados con alta probabilidad de ascenso, facilitando la planificación de recursos humanos.

2. **Personalización de Capacitación:**
   - **Recomendación de Cursos:** La IA puede recomendar cursos específicos basados en el progreso y necesidades individuales de los empleados, optimizando el proceso de capacitación.

3. **Optimización de Recursos:**
   - **Automatización de Recomendaciones de Roles y Promociones:** La IA puede automatizar la recomendación de roles y promociones, reduciendo la carga administrativa y mejorando la eficiencia de la gestión de personal.

4. **Generación de Reportes:**
   - **Reportes Personalizados:** La generación automática de reportes permite una revisión rápida y detallada del estado de capacitación y promociones, facilitando la toma de decisiones informadas.
""")

# 1. Integración de Bases de Datos Reales
st.header("Integración de Bases de Datos Reales")

st.subheader("Datos de Empleados Ficticios")
st.dataframe(df_empleados)

# Filtrado Dinámico de Empleados
st.subheader("Filtrar Empleados")
subdireccion_filter = st.selectbox("Seleccionar Subdirección", df_empleados["Subdirección"].unique())
puesto_filter = st.multiselect("Seleccionar Puesto", df_empleados["Puesto Actual"].unique())

df_filtrado = df_empleados[(df_empleados["Subdirección"] == subdireccion_filter) & 
                            (df_empleados["Puesto Actual"].isin(puesto_filter))]

st.dataframe(df_filtrado)

# 2. Simulación Avanzada
st.header("Simulación Avanzada")

st.subheader("Distribución del Progreso en Capacitación")
progress_option = st.selectbox("Seleccionar Progreso", 
                               ["Progreso IA Básico (%)", "Progreso Python (%)", "Progreso Integración Sistemas (%)"])

fig_bar = px.bar(df_empleados, x='Nombre', y=progress_option, color='Subdirección', 
                 title=f'Distribución del {progress_option}')
st.plotly_chart(fig_bar)

st.subheader("Simulación de Necesidades de Capacitación")
# Simulación simple basada en el progreso actual
df_empleados["Necesidad de Capacitación"] = df_empleados[
    ["Progreso IA Básico (%)", "Progreso Python (%)", "Progreso Integración Sistemas (%)"]
].mean(axis=1).apply(lambda x: "Alta" if x < 70 else "Media" if x < 85 else "Baja")

fig_sim = px.pie(df_empleados, names='Necesidad de Capacitación', title='Simulación de Necesidades de Capacitación')
st.plotly_chart(fig_sim)

# Simulación de Impacto de Capacitación
st.subheader("Simulación de Impacto de Capacitación")
incremento = st.slider("Incrementar Progreso en (%)", 0, 20, 5)

df_empleados["Progreso IA Básico (%)"] += incremento
df_empleados["Progreso Python (%)"] += incremento
df_empleados["Progreso Integración Sistemas (%)"] += incremento

df_empleados["Nueva Necesidad de Capacitación"] = df_empleados[
    ["Progreso IA Básico (%)", "Progreso Python (%)", "Progreso Integración Sistemas (%)"]
].mean(axis=1).apply(lambda x: "Alta" if x < 70 else "Media" if x < 85 else "Baja")

fig_sim_impact = px.pie(df_empleados, names='Nueva Necesidad de Capacitación', 
                        title='Impacto de Incremento en Capacitación')
st.plotly_chart(fig_sim_impact)

# Análisis de Correlación
st.subheader("Análisis de Correlación")
corr = df_empleados[["Progreso IA Básico (%)", "Progreso Python (%)", "Progreso Integración Sistemas (%)"]].corr()

fig_corr = px.imshow(corr, text_auto=True, title="Matriz de Correlación entre Capacidades")
st.plotly_chart(fig_corr)

# 3. Automatización
st.header("Automatización")

st.subheader("Recomendación Automática de Roles y Cursos")
def recomendar_curso(progreso):
    if progreso < 70:
        return "Curso IA Avanzado"
    elif progreso < 85:
        return "Curso Python Avanzado"
    else:
        return "Curso de Integración de Sistemas"

df_empleados["Curso Recomendado"] = df_empleados["Progreso IA Básico (%)"].apply(recomendar_curso)
st.write("Recomendaciones de Cursos:")
st.dataframe(df_empleados[["Nombre", "Curso Recomendado"]])

st.subheader("Recomendación de Promociones")
df_empleados["Recomendación de Promoción"] = df_empleados["Puesto Actual"].apply(
    lambda x: "Gerente" if x == "Coordinador" else "Director" if x == "Gerente" else "Sin promoción"
)
st.write("Recomendaciones de Promoción:")
st.dataframe(df_empleados[["Nombre", "Puesto Actual", "Recomendación de Promoción"]])

# Predicción de Ascensos
st.subheader("Predicción de Ascensos")

# Variables predictoras
X = df_empleados[["Progreso IA Básico (%)", "Progreso Python (%)", "Progreso Integración Sistemas (%)"]]
# Variable objetivo (0: No promoción, 1: Promoción)
y = df_empleados["Recomendación de Promoción"].apply(lambda x: 1 if x != "Sin promoción" else 0)

# Modelo simple
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Predicciones
df_empleados["Probabilidad de Ascenso"] = model.predict_proba(X)[:, 1] * 100

st.dataframe(df_empleados[["Nombre", "Probabilidad de Ascenso"]])

# Generación de Reportes Personalizados
st.subheader("Generar Reporte Personalizado")

buffer = io.BytesIO()

with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
    df_empleados.to_excel(writer, index=False)

buffer.seek(0)  # Importante para volver al inicio del buffer

st.download_button(label="Descargar Reporte Excel",
                   data=buffer,
                   file_name="reporte_empleados.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


st.sidebar.write("© 2024 jahoperi")
