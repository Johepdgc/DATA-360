import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt
import time
import os
import seaborn as sns
from fpdf import FPDF

# Set your Google Gemini API key from environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ No se encontró la clave de API. Define GEMINI_API_KEY como variable de entorno.")
genai.configure(api_key=GEMINI_API_KEY)

# Categories and subcategories
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

def gemini_classify(comment): # Renamed function
    # Choose a Gemini model
    # For text generation tasks like classification, 'gemini-1.5-flash-latest' is a good balance of capability and speed.
    # You can also use 'gemini-pro'.
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    prompt_parts = [
        "Eres un asistente que clasifica quejas de clientes en categorías y subcategorías bien definidas.",
        "Las categorías principales son: " + ", ".join(main_labels) + ".",
        "Cada categoría tiene estas subcategorías:",
        "\n".join([f"{cat}: {', '.join(subs)}" for cat, subs in categories.items()]),
        "\nPara el siguiente comentario, responde SOLO con el nombre de la categoría principal y la subcategoría más adecuada, separados por coma. Ejemplo: Pagos, Cobro duplicado",
        f"\nComentario: {comment}",
        "Categoría, Subcategoría:"
    ]

    try:
        # Configuration for deterministic output and limiting token usage
        generation_config = genai.types.GenerationConfig(
            temperature=0,
            max_output_tokens=60 # Increased to allow for longer category/subcategory names
        )
        response = model.generate_content(prompt_parts, generation_config=generation_config)
        text = response.text.strip()
        parts = [x.strip() for x in text.split(",")]
        if len(parts) == 2:
            # Validate if the returned category and subcategory are known, otherwise default to "Otro"
            main_cat_candidate = parts[0]
            sub_cat_candidate = parts[1]
            if main_cat_candidate in categories and sub_cat_candidate in categories.get(main_cat_candidate, []):
                return pd.Series([main_cat_candidate, sub_cat_candidate])
            else:
                # If Gemini hallucinates a category/subcategory not in our list, try to find best main category
                # and default subcategory, or just default to "Otro"
                if main_cat_candidate in categories:
                     return pd.Series([main_cat_candidate, categories[main_cat_candidate][0] if categories[main_cat_candidate] else "Sin clasificación específica"])
                return pd.Series(["Otro", "Sin clasificación específica"])
        else:
            return pd.Series(["Otro", "Sin clasificación específica"])
    except Exception as e:
        print(f"Error during Gemini API call for comment '{comment[:50]}...': {e}")
        # Check for specific API errors if needed, e.g., rate limits
        # if "rate limit" in str(e).lower():
        #     print("Rate limit likely exceeded. Consider increasing sleep time or checking quotas.")
        return pd.Series(["Otro", "Sin clasificación específica"])

# Load data
df = pd.read_csv("/Users/johepgradis/Downloads/DATA-360/Tracking de solicitudes (Responses) - Form Responses 1.csv", encoding='utf-8', engine='python')
df = df[df['Comentarios'].notna()].copy()

# Filter for 2024 and later (if you added this previously)
df['Marca temporal'] = pd.to_datetime(df['Marca temporal'], errors='coerce', dayfirst=False)
df = df[df['Marca temporal'].dt.year >= 2024]


# Apply Gemini classification (with delay to be mindful of API quotas)
results = []
print(f"Starting classification for {len(df)} comments using Gemini API...")
for i, comment in enumerate(df['Comentarios']):
    if i > 0 and i % 50 == 0: # Print progress every 50 comments
        print(f"Processed {i}/{len(df)} comments...")
    results.append(gemini_classify(comment))
    time.sleep(1)  # Adjust based on Gemini API rate limits (e.g., gemini-1.5-flash default is 60 QPM)

df[['Categoría', 'Subcategoría']] = results
print("Classification complete.")

# Generar resumen por categoría y subcategoría
resumen = df.groupby(['Categoría', 'Subcategoría']).size().reset_index(name='Cantidad')
resumen = resumen.sort_values(['Categoría', 'Cantidad'], ascending=[True, False])
resumen.to_excel("resumen_categorias_gemini.xlsx", index=False)
print("📊 Resumen exportado como resumen_categorias_gemini.xlsx")

# Visualization with Seaborn
plt.figure(figsize=(10, 6))
sns.countplot(y='Categoría', data=df, order=df['Categoría'].value_counts().index, palette='Set2')
plt.title('Distribución de Categorías de Quejas (Gemini)')
plt.xlabel('Cantidad')
plt.ylabel('Categoría')
plt.tight_layout()
plt.savefig("distribucion_categorias_gemini.png")
plt.show()

# Crear reporte en PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Reporte de Clasificación de Quejas (Gemini)", ln=True, align='C')
pdf.ln(10)
pdf.image("distribucion_categorias_gemini.png", x=10, y=30, w=180)
pdf.ln(105)

pdf.set_font("Arial", 'B', 12)
pdf.cell(200, 10, txt="Resumen de Subcategorías", ln=True)
pdf.set_font("Arial", 'B', 10)
pdf.cell(60, 8, "Categoría", 1)
pdf.cell(80, 8, "Subcategoría", 1)
pdf.cell(30, 8, "Cantidad", 1)
pdf.ln()
pdf.set_font("Arial", size=10)

top_subcats = resumen.head(10)
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

pdf.output("reporte_quejas_gemini.pdf")
print("📝 Reporte PDF generado como reporte_quejas_gemini.pdf")
