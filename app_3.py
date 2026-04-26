import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter
from skimage import measure
import geopandas as gpd
from shapely.geometry import Polygon
from fpdf import FPDF

st.set_page_config(layout="wide", page_title="Teste_NDVI_Automático_exporta_QGIS")

st.markdown("""
<style>
.metric-box {
    background-color: #111827;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("🌱 AGREEL")
st.subheader("Análise inteligente de vigor vegetativo (NDVI)")

st.sidebar.header("Insira suas imagens")

red_file = st.sidebar.file_uploader("Banda RED", type=["png", "jpg", "jpeg", "tif", "tiff"])
nir_file = st.sidebar.file_uploader("Banda NIR", type=["png", "jpg", "jpeg", "tif", "tiff"])

def load_image(file):
    img = Image.open(file).convert("L")
    return np.array(img).astype("float32")

def calculate_ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-10)

def classify_kmeans(ndvi):
    h, w = ndvi.shape
    X = ndvi.reshape(-1, 1)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    centers = kmeans.cluster_centers_.flatten()
    order = np.argsort(centers)

    classes = np.zeros_like(labels)
    classes[labels == order[0]] = 1
    classes[labels == order[1]] = 2
    classes[labels == order[2]] = 3

    return classes.reshape(h, w)

def generate_polygons(classes):
    polygons = []

    for value in [1, 2, 3]:
        mask = (classes == value).astype(np.uint8)
        contours = measure.find_contours(mask, 0.5)

        for contour in contours:
            poly = Polygon(contour)

            if poly.is_valid:
                polygons.append({
                    "geometry": poly,
                    "class": value
                })

    gdf = gpd.GeoDataFrame(polygons)

    # Limpeza automática: remove polígonos pequenos
    gdf = gdf[gdf.geometry.area > 1000]

    return gdf

if red_file and nir_file:

    red = load_image(red_file)
    nir = load_image(nir_file)

    if red.shape != nir.shape:
        st.error("As imagens precisam ter o mesmo tamanho.")
    else:

        ndvi = calculate_ndvi(nir, red)
        ndvi_smooth = gaussian_filter(ndvi, sigma=2)

        classes = classify_kmeans(ndvi_smooth)
        gdf = generate_polygons(classes)

        area_critica = np.sum(classes == 1) / classes.size * 100
        st.markdown("## 📊 Análise de Vigor")
        baixo = np.mean(classes == 1) * 100
        medio = np.mean(classes == 2) * 100
        alto = np.mean(classes == 3) * 100

        # =========================
        # CARDS
        # =========================
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        col1.metric("NDVI médio", f"{np.mean(ndvi):.2f}")
        col2.metric("Baixo vigor", f"{baixo:.1f}%")
        col3.metric("Médio vigor", f"{medio:.1f}%")
        col4.metric("Alto vigor", f"{alto:.1f}%")
        col5.metric("Polígonos limpos", len(gdf))
        col6.metric("Pixels críticos", int(np.sum(classes == 1)))

        st.divider()

        # =========================
        # CORES
        # =========================
        cmap_zonas = ListedColormap(["red", "yellow", "green"])

        # =========================
        # MAPAS
        # =========================
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Mapa NDVI")
            fig, ax = plt.subplots()
            im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
            ax.axis("off")
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)

        with c2:
            st.subheader("Zonas de manejo")
            fig2, ax2 = plt.subplots()
            im2 = ax2.imshow(classes, cmap=cmap_zonas, vmin=1, vmax=3)
            ax2.axis("off")
            plt.colorbar(im2, ax=ax2, ticks=[1, 2, 3])
            st.pyplot(fig2)

        # =========================
        # LEGENDA
        # =========================
        st.markdown("""
        ### Legenda das zonas
        🔴 **Baixo vigor** — área crítica  
        🟡 **Médio vigor** — área de atenção  
        🟢 **Alto vigor** — área saudável  
        """)

        st.divider()

        # =========================
        # RELATÓRIO AGRONÔMICO
        # =========================
        if area_critica > 20:
            texto_relatorio = (
                "Alto nível de estresse vegetativo detectado. A proporção de áreas com baixo vigor é elevada, "
                "indicando possível deficiência hídrica, nutricional ou fitossanitária. "
                "Recomenda-se intervenção imediata com diagnóstico em campo e ajuste de manejo."
            )
            st.error(texto_relatorio)

        elif area_critica > 10:
            texto_relatorio = (
                "Variabilidade espacial moderada identificada. A presença de zonas com baixo vigor sugere "
                "heterogeneidade no desenvolvimento da cultura. "
                "Recomenda-se monitoramento contínuo e validação em campo, com foco em solo, irrigação e nutrição."
            )
            st.warning(texto_relatorio)

        else:
            texto_relatorio = (
                "Padrão vegetativo homogêneo observado. A área apresenta predominância de médio a alto vigor, "
                "indicando bom estado fisiológico da cultura. "
                "Manter manejo atual e monitoramento periódico."
            )
            st.success(texto_relatorio)
        st.divider()

        st.markdown("### 📌 Insight estratégico")

        if area_critica > 20:
            st.markdown("➡️ Priorizar manejo localizado (taxa variável) nas zonas críticas.")
        elif area_critica > 10:
            st.markdown("➡️ Implementar monitoramento por talhão e validar com amostragem de solo.")
        else:
            st.markdown("➡️ Manter estratégia atual e acompanhar evolução temporal via NDVI.")

        # =========================
        # DOWNLOAD GEOJSON
        # =========================
        st.markdown("## 📥 Exportação")
        st.subheader("Download dos resultados")

        if len(gdf) > 0:
            geojson = gdf.to_json()

            st.download_button(
                label="📥 Baixar zonas de manejo (GeoJSON)",
                data=geojson,
                file_name="zonas_manejo_agreel.geojson",
                mime="application/geo+json"
            )

            def limpar_texto_pdf(texto):
                return (
                    texto.replace("⚠️", "")
                    .replace("✅", "")
                    .replace("ç", "c")
                    .replace("ã", "a")
                    .replace("õ", "o")
                    .replace("á", "a")
                    .replace("é", "e")
                    .replace("í", "i")
                    .replace("ó", "o")
                    .replace("ú", "u")
                )

        # =========================
        # DOWNLOAD PDF
        # =========================
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="AGREEL PRO - Relatório Agronômico", ln=True, align='C')

        pdf.set_font("Arial", "", 12)
        pdf.ln(5)
        pdf.cell(0, 10, f"NDVI medio: {np.mean(ndvi):.2f}", ln=True)
        pdf.cell(0, 10, f"Baixo vigor: {baixo:.1f}%", ln=True)
        pdf.cell(0, 10, f"Medio vigor: {medio:.1f}%", ln=True)
        pdf.cell(0, 10, f"Alto vigor: {alto:.1f}%", ln=True)
        pdf.cell(0, 10, f"Area critica: {area_critica:.1f}%", ln=True)
        pdf.cell(0, 10, f"Poligonos gerados apos limpeza: {len(gdf)}", ln=True)

        pdf.ln(5)
        pdf.multi_cell(0, 10, limpar_texto_pdf(texto_relatorio))

        texto_pdf = limpar_texto_pdf(texto_relatorio)

        pdf_bytes = pdf.output(dest="S").encode("latin-1")

        pdf.multi_cell(0, 10, texto_pdf)

        st.download_button(
            label="📄 Baixar relatório PDF",
            data=pdf_bytes,
            file_name=f"relatorio_agreel_{int(area_critica)}pct.pdf",
            mime="application/pdf"
        )

else:
    st.info("Faça upload das bandas RED e NIR.")
