import cv2
import matplotlib.pyplot as plt

# Capturar a imagem da linha de produção
cap = cv2.imread("Ferramentas-de-pedreiro-capa.jpg")


# Aplicar um filtro de suavização na imagem
blurred = cv2.GaussianBlur(cap, (5, 5), 0)

# Converter a imagem para escala de cinza
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

# Aplicar um threshold na imagem
_, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

# Aplicar a operação de abertura na imagem binarizada
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Encontrar os contornos das peças
contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Verificar a área de cada contorno e determinar se uma peça está presente ou ausente
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 1000 and area < 5000:
        # Peça presente na linha de produção
        cv2.drawContours(cap, [contour], 0, (0, 255, 0), 2)
    else:
        # Peça ausente na linha de produção
        cv2.drawContours(cap, [contour], 0, (0, 0, 255), 2)

# Mostrar a imagem resultante
# cv2.imshow("Produção", cap)

# a imagem com cores trocadas por causa do BGR do open_cv
plt.imshow(cap)

plt.show()