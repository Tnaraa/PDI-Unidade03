#Capítulo 21. Extração de contornos
#exercicio Modifique o programa para extrair os contornos internos das componentes conectadas presentes na imagem formas.png
import cv2

# Carregar a imagem em escala de cinza
image = cv2.imread('input/formas.png', cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Não abriu image.jpg")
else:

    # Aplicar limiarização de Otsu
    _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #Pontos com CHAIN_APPROX_NONE
    # Encontrar contornos
    contorno, hi = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    total_pointsnone = sum(len(cont) for cont in contorno)
    print(f"Total de pontos nos contornos usando CHAIN_APPROX_NONE: {total_pointsnone}")

    # Pontos com CHAIN_APPROX_SIMPLE
    # Encontrar contornos
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    total_points = sum(len(contour) for contour in contours)
    print(f"Total de pontos nos contornos com CHAIN_APPROX_SIMPLE: {total_points}")

    # Converter imagem para BGR (para visualização colorida)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Abrir arquivo para salvar os contornos em formato SVG
    try:
        with open("output/contornos2.svg", "w") as file:
            file.write(f'<svg height="{image.shape[0]}" width="{image.shape[1]}" xmlns="http://www.w3.org/2000/svg">\n')

            for contour in contours:

                file.write(f'<path d="M {contour[0][0][0]} {contour[0][0][1]} ')
                for point in contour[1:]:
                    file.write(f'L {point[0][0]} {point[0][1]} ')
                file.write('Z" fill="#cccccc" stroke="black" stroke-width="1" />\n')
                #pontosSimples += len(contours[i])
                # Desenhar os contornos na imagem
                cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)

            file.write('</svg>\n')

    except IOError:
        print("Não abriu contornos.svg")

    # Mostrar a imagem com os contornos desenhados
    cv2.imshow("Imagem", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()