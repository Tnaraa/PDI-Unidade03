import cv2
from math import copysign, log10, sqrt

# Função para calcular os momentos de Hu de uma imagem
def calculate_hu_moments(image):
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments_transformed = [
        -1 * copysign(1.0, h) * log10(abs(h)) if h != 0 else 0 for h in hu_moments
    ]
    return hu_moments_transformed

# Função para calcular a distância euclidiana entre dois vetores
def calculate_euclidean_distance(vec1, vec2):
    return sqrt(sum((v1 - v2) ** 2 for v1, v2 in zip(vec1, vec2)))

scale_factor=0.5
# Carregar imagens em escala de cinza
person_image = cv2.imread("input/pessoa.jpg", cv2.IMREAD_GRAYSCALE)
crowd_image = cv2.imread("input/multidao.jpg", cv2.IMREAD_GRAYSCALE)
print(crowd_image.shape)
if person_image is None or crowd_image is None:
    print("Não foi possível carregar as imagens!")
    exit()

# Redimensionar imagens para acelerar o processamento
person_image = cv2.resize(person_image, (0, 0), fx=scale_factor, fy=scale_factor)
crowd_image = cv2.resize(crowd_image, (0, 0), fx=scale_factor, fy=scale_factor)

# Calcular os momentos de Hu da imagem da pessoa
person_hu_moments = calculate_hu_moments(person_image)

person_rows, person_cols = person_image.shape
crowd_rows, crowd_cols = crowd_image.shape

min_distance = float("inf")
best_match = None

# Deslizar a janela sobre a imagem da multidão
for y in range(crowd_rows - person_rows + 1):
    for x in range(crowd_cols - person_cols + 1):
        roi = crowd_image[y: y + person_rows, x: x + person_cols]
        roi_hu_moments = calculate_hu_moments(roi)
        distance = calculate_euclidean_distance(person_hu_moments, roi_hu_moments)

        if distance < min_distance:
            min_distance = distance
            print(distance)
            best_match = (x, y, person_cols, person_rows)

# Garantir que foi encontrada uma correspondência
if best_match:
    x, y, w, h = best_match
    print(f"Melhor correspondência encontrada na posição: {x}, {y}, tamanho: {w}x{h}, distância: {min_distance:.6f}")

    # Converter a imagem para BGR para desenhar em cores
    result_image = cv2.cvtColor(crowd_image, cv2.COLOR_GRAY2BGR)

    # Desenhar o retângulo na melhor correspondência
    result_image = cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 4)
    result_image = cv2.resize(result_image, (1500, 900), interpolation=cv2.INTER_LINEAR)
    # Exibir a imagem resultante
    cv2.imshow("Melhor Correspondência", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Nenhuma correspondência encontrada!")


