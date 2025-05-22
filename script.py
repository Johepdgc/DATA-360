import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
from openai import OpenAI 
import os

# 1. Load data
df = pd.read_csv("/Users/johepgradis/Downloads/DATA-360/Tracking de solicitudes (Responses) - Form Responses 1.csv", encoding='utf-8', engine='python')
df = df[df['Comentarios'].notna()].copy()

# Embeddings y agrupamiento de comentarios
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['Comentarios'].tolist(), show_progress_bar=True)

# Aplicar clustering
n_clusters = 8  # puedes ajustar este valor según los resultados
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

# Configurar clave de API de OpenAI desde variable de entorno
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # <-- Initialize client
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("❌ No se encontró la clave OPENAI_API_KEY. Debes definirla como variable de entorno.")

df['Cluster'] = cluster_labels

cluster_descriptions = {}
for cluster_id in sorted(df['Cluster'].unique()):
    ejemplos = df[df['Cluster'] == cluster_id]['Comentarios'].head(5).tolist()
    prompt = (
        "Estos son ejemplos de quejas de usuarios que pertenecen a un mismo grupo. "
        "Resume en una sola frase el tema principal que los une:\n\n"
        + "\n".join(f"- {ej}" for ej in ejemplos)
    )

    try:
        response = client.chat.completions.create(  # <-- Use new client here
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "Eres un analista de experiencia de cliente que resume tipos de quejas de usuarios."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        descripcion = response.choices[0].message.content.strip()
    except Exception as e:
        descripcion = "Descripción no disponible"
        print(f"Error con cluster {cluster_id}: {e}")

    cluster_descriptions[cluster_id] = descripcion

# Agregar nombres de cluster al DataFrame
df['Nombre_Grupo'] = df['Cluster'].map(cluster_descriptions)

# Parse 'Marca temporal' as datetime
df['Marca temporal'] = pd.to_datetime(df['Marca temporal'], errors='coerce', dayfirst=False)

# Filter for 2024 and later
df = df[df['Marca temporal'].dt.year >= 2024]

# 2. Define categories and subcategories
categories = {
    "Pagos": ["Cobro duplicado", "Tarjeta rechazada", "Comisión inesperada", "Otros problemas de pago"],
    "Reembolso": ["Retraso de reembolso", "Reembolso no recibido", "Otros reembolsos"],
    "Entradas": ["Transferencia de entradas", "Entrada no recibida", "Cantidad incorrecta", "Otros problemas de entradas"],
    "Soporte Técnico": ["Problemas de acceso", "Fallo de plataforma", "Otros errores técnicos"],
    "Solicitud de Información": ["Dónde comprar", "Precios y costos", "Horarios", "Otras consultas"],
    "Error en la plataforma": ["Problemas con links de pago", "Problemas con tickets", "Otros errores"],
    "Afiliación": ["Solicitud de afiliación", "Problemas con afiliación"],
    "Otro": ["Sin clasificación específica"]
}

main_labels = list(categories.keys())
sub_labels = [sub for subs in categories.values() for sub in subs]

# 3. Load zero-shot classifier (multilingual)
classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli", use_fast=False)

CATEGORIA_COL = 'Categoría'

def classify_comment(comment):
    main_result = classifier(comment, main_labels, multi_label=False)
    main_cat = main_result['labels'][0]
    main_score = main_result['scores'][0]

    sub_result = classifier(comment, categories[main_cat], multi_label=False)
    sub_cat = sub_result['labels'][0]
    sub_score = sub_result['scores'][0]

    return pd.Series([main_cat, main_score, sub_cat, sub_score])

# 4. Apply classification
df[['Categoría', 'Confianza_Categoría', 'Subcategoría', 'Confianza_Subcategoría']] = df['Comentarios'].apply(classify_comment)

# Visualización con Seaborn
plt.figure(figsize=(10, 6))
sns.countplot(y='Categoría', data=df, order=df['Categoría'].value_counts().index, palette='viridis')
plt.title('Distribución de Categorías de Quejas')
plt.xlabel('Cantidad')
plt.ylabel('Categoría')
plt.tight_layout()
plt.savefig("distribucion_categorias.png")
plt.show()

# 6. Save results
df.to_csv("resultados_clasificados_ml.csv", index=False)
print("✅ Clasificación ML completada: resultados_clasificados_ml.csv")

# Guardar archivo agrupado por clusters
df.to_csv("resultados_clasificados_cluster.csv", index=False)
print("✅ Agrupamiento semántico completado: resultados_clasificados_cluster.csv")


# Generar resumen por categoría y subcategoría y exportar a Excel
resumen = df.groupby(['Categoría', 'Subcategoría']).size().reset_index(name='Cantidad')
resumen = resumen.sort_values(['Categoría', 'Cantidad'], ascending=[True, False])
resumen.to_excel("resumen_categorias.xlsx", index=False)
print("📊 Resumen exportado como resumen_categorias.xlsx")

# Crear reporte en PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Reporte de Clasificación de Quejas", ln=True, align='C')
pdf.ln(10)
pdf.image("distribucion_categorias.png", x=10, y=30, w=180)
pdf.ln(105)

pdf.set_font("Arial", 'B', 12)
pdf.cell(200, 10, txt="Resumen de Subcategorías", ln=True)
pdf.set_font("Arial", size=10)

# Mostrar las 10 subcategorías más frecuentes en formato de tabla
top_subcats = resumen.head(10)
pdf.set_font("Arial", 'B', 10)
pdf.cell(60, 8, "Categoría", 1)
pdf.cell(80, 8, "Subcategoría", 1)
pdf.cell(30, 8, "Cantidad", 1)
pdf.ln()

pdf.set_font("Arial", size=10)
for _, row in top_subcats.iterrows():
    pdf.cell(60, 8, str(row['Categoría']), 1)
    pdf.cell(80, 8, str(row['Subcategoría']), 1)
    pdf.cell(30, 8, str(row['Cantidad']), 1)
    pdf.ln()

pdf.ln(10)
pdf.set_font("Arial", 'B', 12)
pdf.cell(200, 10, txt="Comentarios del Análisis:", ln=True)
pdf.set_font("Arial", size=10)
comentario = f"La mayoría de las quejas registradas pertenecen a la categoría '{df['Categoría'].value_counts().idxmax()}', siendo la subcategoría más frecuente '{df['Subcategoría'].value_counts().idxmax()}'. Se recomienda enfocar esfuerzos en resolver esta problemática primero."
pdf.multi_cell(0, 8, comentario)

pdf.output("reporte_quejas.pdf")
print("📝 Reporte PDF generado como reporte_quejas.pdf")

# Visualización con gráfico de pastel para clusters
cluster_counts = df['Nombre_Grupo'].value_counts()
plt.figure(figsize=(9, 9))
plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Distribución de Temas Detectados (Nombres Automáticos)")
plt.tight_layout()
plt.savefig("grafico_pastel_clusters.png")
plt.show()

# Incluir el gráfico y comentario del grupo más frecuente en el PDF
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Análisis de Temas por Agrupamiento Semántico", ln=True, align='C')
pdf.ln(10)
pdf.image("grafico_pastel_clusters.png", x=10, y=30, w=180)
pdf.ln(105)
pdf.set_font("Arial", 'B', 12)
pdf.cell(200, 10, txt="Grupo más frecuente detectado:", ln=True)
pdf.set_font("Arial", size=10)
top_named = cluster_counts.idxmax()
ejemplo = df[df['Nombre_Grupo'] == top_named]['Comentarios'].iloc[0]
pdf.multi_cell(0, 8, f"El grupo más común detectado es: {top_named}.\n\nEjemplo de queja en este grupo:\n\"{ejemplo}\"")
