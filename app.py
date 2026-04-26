import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(layout="wide")

st.title("🌱 CotTech Agro IA")
st.subheader("Análise de vigor vegetativo (NDVI)")

st.sidebar.header("Upload das bandas")

red_file = st.sidebar.file_uploader("Banda RED", type=["png", "jpg", "tif"])
nir_file = st.sidebar.file_uploader("Banda NIR", type=["png", "jpg", "tif"])

def load_image(file):
    img = Image.open(file).convert("L")
    return np.array(img).astype("float32")

def calculate_ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-10)

def classify(ndvi):
    classes = np.zeros_like(ndvi)

    classes[ndvi < 0.3] = 1
    classes[(ndvi >= 0.3) & (ndvi < 0.6)] = 2
    classes[ndvi >= 0.6] = 3

    return classes

if red_file and nir_file:

    red = load_image(red_file)
    nir = load_image(nir_file)

    if red.shape != nir.shape:
        st.error("As imagens precisam ter o mesmo tamanho")
    else:

        ndvi = calculate_ndvi(nir, red)
        classes = classify(ndvi)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("NDVI médio", f"{np.mean(ndvi):.2f}")
        col2.metric("Baixo vigor", f"{np.mean(classes==1)*100:.1f}%")
        col3.metric("Médio vigor", f"{np.mean(classes==2)*100:.1f}%")
        col4.metric("Alto vigor", f"{np.mean(classes==3)*100:.1f}%")

        st.divider()

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
            im2 = ax2.imshow(classes, cmap="RdYlGn", vmin=1, vmax=3)
            ax2.axis("off")
            plt.colorbar(im2, ax=ax2)
            st.pyplot(fig2)

        st.divider()

        st.subheader("Diagnóstico")

        media = np.mean(ndvi)

        if media > 0.6:
            st.success("Lavoura com alto vigor vegetativo.")
        elif media > 0.3:
            st.warning("Lavoura com vigor intermediário. Monitorar áreas.")
        else:
            st.error("Baixo vigor detectado. Priorizar vistoria de campo.")

else:
    st.info("Faça upload das bandas RED e NIR.")