import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(
    page_title="CotTech Agro IA",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 CotTech Agro IA")
st.subheader("Análise inicial de vigor vegetativo por NDVI")

st.markdown("""
Sistema inteligente para análise de imagens agrícolas, geração de mapas de vigor,
zonas de manejo e relatório agronômico.
""")

st.sidebar.header("Entrada de dados")

red_file = st.sidebar.file_uploader("Upload da banda RED", type=["tif", "tiff", "png", "jpg"])
nir_file = st.sidebar.file_uploader("Upload da banda NIR", type=["tif", "tiff", "png", "jpg"])

def load_image(file):
    image = Image.open(file).convert("L")
    return np.array(image).astype("float32")

def calculate_ndvi(nir, red):
    ndvi = (nir - red) / (nir + red + 1e-10)
    return np.clip(ndvi, -1, 1)

def classify_ndvi(ndvi):
    classes = np.zeros_like(ndvi)

    classes[ndvi < 0.30] = 1       # baixo vigor
    classes[(ndvi >= 0.30) & (ndvi < 0.60)] = 2  # médio vigor
    classes[ndvi >= 0.60] = 3      # alto vigor

    return classes

if red_file and nir_file:
    red = load_image(red_file)
    nir = load_image(nir_file)

    if red.shape != nir.shape:
        st.error("As bandas RED e NIR precisam ter o mesmo tamanho.")
    else:
        ndvi = calculate_ndvi(nir, red)
        classes = classify_ndvi(ndvi)

        ndvi_mean = np.nanmean(ndvi)
        low_area = np.sum(classes == 1) / classes.size * 100
        medium_area = np.sum(classes == 2) / classes.size * 100
        high_area = np.sum(classes == 3) / classes.size * 100

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("NDVI médio", f"{ndvi_mean:.2f}")
        col2.metric("Baixo vigor", f"{low_area:.1f}%")
        col3.metric("Médio vigor", f"{medium_area:.1f}%")
        col4.metric("Alto vigor", f"{high_area:.1f}%")

        st.divider()

        col_map1, col_map2 = st.columns(2)

        with col_map1:
            st.subheader("Mapa NDVI")
            fig, ax = plt.subplots()
            im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)

        with col_map2:
            st.subheader("Zonas de vigor")
            fig2, ax2 = plt.subplots()
            im2 = ax2.imshow(classes, cmap="RdYlGn", vmin=1, vmax=3)
            ax2.axis("off")
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            st.pyplot(fig2)

        st.divider()

        st.subheader("Relatório agronômico preliminar")

        if ndvi_mean >= 0.60:
            diagnostico = "A lavoura apresenta predominância de alto vigor vegetativo."
            recomendacao = "Manter o monitoramento e comparar com imagens futuras para avaliar estabilidade espacial."
        elif ndvi_mean >= 0.30:
            diagnostico = "A lavoura apresenta vigor intermediário, com possíveis áreas de atenção."
            recomendacao = "Recomenda-se vistoria em campo nas regiões de médio e baixo vigor."
        else:
            diagnostico = "A lavoura apresenta baixo vigor médio, indicando possível estresse ou falhas na área."
            recomendacao = "Priorizar inspeção agronômica nas zonas críticas para verificar água, nutrição, pragas, doenças ou falhas de plantio."

        st.write(f"**Diagnóstico:** {diagnostico}")
        st.write(f"**Recomendação:** {recomendacao}")

else:
    st.info("Faça upload das bandas RED e NIR para iniciar a análise.")