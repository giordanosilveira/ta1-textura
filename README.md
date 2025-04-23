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

```plain text
├── .gitignore
├── README.md
├── content
    ├── imagens_urbanas
    │   ├── 2017-10-27-06-39-01.jpg
    │   ├── 62cfb95587db43b5e78992f89ff165b2.jpg
    │   ├── 898ea2324d613bf3858c613315937fc8.jpg
    │   ├── 9708454329bd7f8b9777624bcfee6d5f.jpg
    │   ├── Casas-grandes-edifícios-pontes-viadutos-são-alguns-dos-elementos-que-compõem-a-paisagem-urbana.-Fonte-Pixabay.jpg
    │   ├── Rurik-1090-1280x720.jpg
    │   ├── a-paisagem-urbana-e-a-matematica-da-cidade_1.jpg
    │   ├── a3da7f1406e9f211a10223edbac0cc6c.jpg
    │   ├── buildings-7109918_1280.jpg
    │   ├── istockphoto-1406960186-612x612.jpg
    │   ├── paisagem-urbana.jpg
    │   ├── paisagem-urbana_405233185-scaled.jpg
    │   ├── pexels-photo-2299949.jpeg
    │   ├── photo-1480714378408-67cf0d13bc1b.jpeg
    │   ├── pngtree-cityscape-charm-a-tale-of-towers-image_15865523.jpg
    │   └── pngtree-new-york-city-manhattan-downtown-skyline-cityscape-evening-landmark-photo-image_24249822.jpg
    └── segmentadas
    │   ├── segmentado_2017-10-27-06-39-01.jpg
    │   ├── segmentado_62cfb95587db43b5e78992f89ff165b2.jpg
    │   ├── segmentado_898ea2324d613bf3858c613315937fc8.jpg
    │   ├── segmentado_9708454329bd7f8b9777624bcfee6d5f.jpg
    │   ├── segmentado_Casas-grandes-edifícios-pontes-viadutos-são-alguns-dos-elementos-que-compõem-a-paisagem-urbana.-Fonte-Pixabay.jpg
    │   ├── segmentado_Rurik-1090-1280x720.jpg
    │   ├── segmentado_a-paisagem-urbana-e-a-matematica-da-cidade_1.jpg
    │   ├── segmentado_a3da7f1406e9f211a10223edbac0cc6c.jpg
    │   ├── segmentado_buildings-7109918_1280.jpg
    │   ├── segmentado_istockphoto-1406960186-612x612.jpg
    │   ├── segmentado_paisagem-urbana.jpg
    │   ├── segmentado_paisagem-urbana_405233185-scaled.jpg
    │   ├── segmentado_pexels-photo-2299949.jpg
    │   ├── segmentado_photo-1480714378408-67cf0d13bc1b.jpg
    │   ├── segmentado_pngtree-cityscape-charm-a-tale-of-towers-image_15865523.jpg
    │   └── segmentado_pngtree-new-york-city-manhattan-downtown-skyline-cityscape-evening-landmark-photo-image_24249822.jpg
└── main.py
```

## 📑 Como executar

1. Crie um ambiente virtual
```bash
python3 -m venv .venv && source .venv/bin/activate
```
2. Instale os recursos necessários
```bash
pip install -r requirements.txt
```
3. Execute o script
```
python main.py
```

## 📁 Output
Tanto o input usado, quanto o resultado estão na pasta `content`. 

As imagens usadas estão na pasta `imagens_urbanas` e as imagens segmentadas, resultados da execução do script, estão na pasta `segmentadas`. 
Para comparar as imagens, veja o nome da imagem original na pasta `imagens_urbanas` e procure o mesmo nome na pasta `segmentadas`


