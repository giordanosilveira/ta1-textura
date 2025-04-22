# Segmentação de Imagens por Textura com Filtros de Gabor

Este projeto tem como objetivo segmentar imagens com base em **texturas**, utilizando **filtros clássicos de Gabor** e um **filtro circular**, processando a imagem em **três escalas**. Os descritores de textura são agrupados utilizando o algoritmo **KMeans**, e a segmentação é exibida visualmente.

## 📌 Objetivo do trabalho

> Escolher um tipo de imagem para segmentar por textura. Criar um conjunto de filtros de textura com orientações horizontal, vertical, 45° e 135°, além de um filtro circular. Processar a imagem em 3 escalas, calcular a média local das respostas dos filtros, agrupar regiões semelhantes com KMeans, e mostrar as segmentações resultantes. O código deve ser aplicado a pelo menos 16 imagens e **não utilizar deep learning**.

---

## ✅ Relação entre requisitos e implementação

| Requisito | Implementação no código |
|----------|--------------------------|
| **Tipo de imagem escolhido** | A pasta `"content/imagens_urbanas"` contém imagens urbanas para segmentação. |
| **Filtros com orientações 0°, 45°, 90° e 135°** | Criados na função `build_gabor_filters()` com `theta = 0, π/4, π/2, 3π/4`. |
| **Filtro circular** | Adicionado na mesma função com um **filtro Laplaciano do Gaussiano 2D**, que detecta texturas isotrópicas. |
| **Três escalas de imagem** | Aplicação de `cv2.pyrDown` em duas etapas, criando três versões da imagem (original, 1/2 e 1/4 da resolução). |
| **Aplicação dos filtros** | `process_with_filters()` realiza convolução entre cada imagem e os filtros. |
| **Vetores com média em janelas** | `compute_texture_features()` calcula a média local da resposta de cada filtro com `cv2.blur(..., (7, 7))`. |
| **Agrupamento por similaridade (KMeans)** | `KMeans(n_clusters=3)` agrupa vetores de textura com base na distância euclidiana. |
| **Visualização da segmentação** | Imagem colorida gerada com `cv2.applyColorMap` e salva com `cv2.imwrite`. |
| **Aplicado a várias imagens (mín. 16)** | Função `process_folder(...)` percorre todos os arquivos `.jpg`, `.png`, `.jpeg` de uma pasta. |
| **Sem uso de deep learning** | O projeto não usa TensorFlow, PyTorch ou redes neurais. Apenas OpenCV, NumPy e Scikit-learn (KMeans). |

---

## 🗂 Estrutura do Projeto

