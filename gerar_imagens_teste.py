import numpy as np
from PIL import Image

# Tamanho da imagem simulada
altura, largura = 600, 900

# Simula variação espacial da lavoura
x = np.linspace(0, 1, largura)
y = np.linspace(0, 1, altura)
xx, yy = np.meshgrid(x, y)

# Banda RED simulada
red = 80 + 60 * xx + 30 * np.random.rand(altura, largura)

# Banda NIR simulada com regiões de vigor diferente
nir = 160 + 80 * (1 - yy) + 40 * np.random.rand(altura, largura)

# Cria uma "mancha" de baixo vigor
mask = (xx - 0.65)**2 + (yy - 0.45)**2 < 0.04
nir[mask] = nir[mask] * 0.45

# Normaliza para 0-255
red_img = np.clip(red, 0, 255).astype(np.uint8)
nir_img = np.clip(nir, 0, 255).astype(np.uint8)

Image.fromarray(red_img).save("banda_RED_teste.png")
Image.fromarray(nir_img).save("banda_NIR_teste.png")

print("Imagens de teste criadas com sucesso.")