import cv2
import math

# Carregar a imagem em escala de cinza
image = cv2.imread("input/formas.png" , cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Erro ao carregar a imagem")
    exit()

# Abrir o arquivo para salvar os momentos
with open("momentos.txt", "w") as file:
    # Limiarização da imagem
    _, thresh_image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

     # Encontrar contornos
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Converter a imagem para BGR para desenhar os contornos em cores
    image_color = cv2.cvtColor(thresh_image, cv2.COLOR_GRAY2BGR)

    nformas = 0

    for i, contour in enumerate(contours):
        if len(contour) < 10:  # Ignorar ruídos ou contornos pequenos
            continue

        nformas += 1

        # Calcular momentos
        momentos = cv2.moments(contour)
        if momentos["m00"] == 0:
            continue

        center_x = int(momentos["m10"] / momentos["m00"])
        center_y = int(momentos["m01"] / momentos["m00"])
        center = (center_x, center_y)

        # Calcular Momentos de Hu
        hu = cv2.HuMoments(momentos).flatten()

        # Transformar os momentos de Hu para escala logarítmica
        for j in range(7):
            hu[j] = -1 * math.copysign(1.0, hu[j]) * math.log10(abs(hu[j]))

        # Desenhar os contornos com base no valor de Hu[0]
        if hu[0] > 0:
            color = (0, 0, 255)  # Vermelho
        else:
            color = (0, 255, 0)  # Verde

        cv2.drawContours(image_color, [contour], -1, color, 2)

        # Rotular o objeto na imagem
        cv2.putText(image_color,str(i),center,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 0),
                4,cv2.LINE_AA,)
        cv2.putText( image_color,str(i),center,cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255),1,cv2.LINE_AA,)

        # Salvar os momentos no arquivo
        file.write(f"Objeto {i}: ")
        file.write(", ".join(f"{value:.6f}" for value in hu))
        file.write("\n")

    print(f"Número de objetos detectados: {nformas}")

# Mostrar a imagem final com contornos e rótulos

cv2.imshow("Contornos e Rótulos", image_color)
cv2.imwrite("contornos-rotulados.png", image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()


