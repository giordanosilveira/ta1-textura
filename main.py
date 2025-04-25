import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

KSIZE = 35  # Tamanho do kernel
SIGMA = 3.0 # Desvio padrão
LAMBDA = 7.0 # Comprimento de onda
GAMMA = 0.5 # Aspecto da elipse
PSI = 0 # Fase

# Criação dos filtros de Gabor para diferentes orientações
def build_gabor_filters():
    filters = []
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    for theta in orientations:
        kernel = cv2.getGaborKernel((KSIZE, KSIZE), SIGMA, theta, LAMBDA, GAMMA, PSI, ktype=cv2.CV_32F)
        filters.append(kernel)

    # Adiciona filtro circular (Laplaciano do Gaussiano)
    log_kernel = cv2.getGaussianKernel(KSIZE, SIGMA)
    log_kernel_2d = log_kernel @ log_kernel.T  # Produto externo para gerar kernel 2D
    log_kernel_2d = log_kernel_2d.astype(np.float32)  # Converte para float32
    laplacian_kernel = cv2.Laplacian(log_kernel_2d, cv2.CV_32F)
    filters.append(laplacian_kernel)

    return filters

# Aplica todos os filtros de Gabor e retorna as respostas
def process_with_filters(img, filters):
    responses = []
    for kern in filters:
        filtered = cv2.filter2D(img, cv2.CV_8UC3, kern)  # Convolução com o filtro
        responses.append(filtered)
    return responses

# Calcula descritores de textura com base nas respostas dos filtros
def compute_texture_features(img, filters):
    responses = process_with_filters(img, filters)
    h, w = img.shape
    feature_vectors = np.zeros((h, w, len(filters)), dtype=np.float32)

    for i, response in enumerate(responses):
        # Média local com janela deslizante 7x7
        mean_response = cv2.blur(response, (7, 7))
        feature_vectors[:, :, i] = mean_response

    return feature_vectors


# Aplica k-means e segmenta a imagem com base nas texturas
def segment_image(img_path, filters, n_clusters=3):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Erro ao carregar imagem: {img_path}")
        return

    # Redimensiona para padronizar
    img = cv2.resize(img, (512, 512))
    
    # Escala 1
    feat1 = compute_texture_features(img, filters).reshape(-1, len(filters))

    # Escala 2
    img2 = cv2.resize(img, (int(512 * 0.66), int(512 * 0.66)))
    feat2 = compute_texture_features(img2, filters).reshape(-1, len(filters))

    # Escala 3
    img3 = cv2.resize(img2, (int(512 * 0.33), int(512 * 0.33)))
    feat3 = compute_texture_features(img3, filters).reshape(-1, len(filters))

    # Concatena as três escalas
    all_features = np.vstack((feat1, feat2, feat3))

    # Agrupamento usando KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    kmeans.fit(all_features)

    # Segmenta a imagem original usando os rótulos da escala 1
    labels = kmeans.predict(feat1)
    segmented = labels.reshape(512, 512).astype(np.uint8)

    # Normaliza e colore para visualização
    segmented_color = cv2.applyColorMap(cv2.convertScaleAbs(segmented * (255 // n_clusters)), cv2.COLORMAP_JET)

    # Salva imagem segmentada
    base = os.path.basename(img_path)
    nome_saida = f"{os.path.splitext(base)[0]}.jpg"
    cv2.imwrite(os.path.join("content/segmentadas", nome_saida), segmented_color)

# Função principal que processa a pasta de imagens
def process_folder(folder_path):
    os.makedirs("content/segmentadas", exist_ok=True)
    filters = build_gabor_filters()

    for file in os.listdir(folder_path):
        if file.endswith((".jpg", ".png", ".jpeg")):
            segment_image(os.path.join(folder_path, file), filters, 5)

# Exemplo de uso
if __name__ == "__main__":
    process_folder("content/imagens_urbanas")