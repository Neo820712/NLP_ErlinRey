# NLP_ErlinRey

## Desafío 1: Vectorización y clasificación con 20 Newsgroups
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

Este desafío implementa un flujo completo de procesamiento de lenguaje natural, desde la representación de texto hasta la clasificación y el análisis semántico, utilizando el dataset *20 Newsgroups*.

![Descripción de la imagen](images/tp1.jpg)


El primer paso consistió en la **vectorización de texto** mediante **tf-idf**. Se realizó un análisis iterativo de la similaridad de documentos (usando similitud coseno), demostrando cómo el filtrado de *stop-words* y el ajuste del hiperparámetro `min_df` (frecuencia mínima de documento) son cruciales para reducir el ruido y mejorar la coherencia semántica de los vectores. Sobre esta representación optimizada, se construyó un clasificador *baseline* de **prototipos (1-nn)**, que sirvió como punto de referencia midiendo la precisión basada puramente en la similitud del vecino más cercano.

La segunda fase se centró en la **clasificación probabilística**. Se entrenaron, compararon y optimizaron modelos **Multinomial Naïve Bayes (MNB)** y **Complement Naïve Bayes (CNB)**. Mediante una búsqueda de hiperparámetros (`alpha` y `min_df`), se buscó maximizar el **f1-score macro**, métrica robusta para este tipo de corpus. El análisis demostró la superioridad de CNB sobre MNB para este dataset, logrando una mejor gestión de las clases con vocabularios solapados. Finalmente, se exploró la **similaridad entre palabras** transponiendo la matriz documento-término; este ejercicio demostró cómo la co-ocurrencia contextual permite agrupar términos semánticamente relacionados (ej. "god" con "jesus", "space" con "nasa"), ilustrando el principio fundamental de la semántica distribucional.
____

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Neo820712/NLP_ErlinRey/blob/main/ErlinRey_Desafio_1.ipynb)
______

## Desafío 2: Word Embeddings y Análisis Semántico con Gensim
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Gensim](https://img.shields.io/badge/Gensim-5B3F88?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

En este desafío se abordó la representación distribuida del lenguaje mediante **word embeddings**, aplicando las metodologías vistas en la Clase 2.  
El objetivo fue construir, visualizar e interpretar representaciones densas de palabras a partir de un corpus temático específico.

![Descripción de la imagen](images/tp2.jpg)

El corpus seleccionado correspondió a letras de canciones de *Michael Jackson*, extraídas del dataset *Songs*.  
Tras un proceso de preprocesamiento (tokenización, normalización y eliminación de stop-words), se entrenaron dos modelos con **Gensim Word2Vec**: **CBOW** (Continuous Bag of Words) y **Skip-Gram**, utilizando ventanas de contexto y muestreo negativo.

La evaluación incluyó el análisis de palabras más y menos similares, donde CBOW mostró asociaciones más frecuentes y contextuales, mientras que Skip-Gram capturó relaciones semánticas más específicas y expresivas.  
Posteriormente, se aplicó una reducción de dimensionalidad (*PCA → t-SNE*) para proyectar los embeddings en 2D, identificando agrupamientos temáticos coherentes con la lírica del artista: amor, baile, reflexión y performance.

Finalmente, se compararon ambos modelos, concluyendo que **CBOW** produce representaciones más estables y generalistas, mientras que **Skip-Gram** discrimina mejor entre contextos y capta matices líricos más sutiles.  
El trabajo evidenció cómo los embeddings logran encapsular la semántica de un corpus a través de la distribución contextual de las palabras, profundizando los conceptos de la semántica distribucional explorados en el Desafío 1.

____

 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Neo820712/NLP_ErlinRey/blob/main/ErlinRey_Desafio_2.ipynb)
______

## Desafío 3: Modelado de Lenguaje con Redes Recurrentes (RNN, GRU, LSTM)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

Este desafío se centró en la construcción y evaluación comparativa de modelos de lenguaje a nivel de caracteres (*Character-Level Language Models*), utilizando arquitecturas de redes neuronales recurrentes implementadas en **PyTorch**.

![Descripción de la imagen](images/tp3.jpg)

El objetivo principal fue entrenar un modelo capaz de aprender la estructura gramatical y el estilo literario de un corpus complejo (*Don Quijote de la Mancha*), para posteriormente generar texto coherente mediante diferentes estrategias de decodificación.

El flujo de trabajo incluyó:
1.  **Preprocesamiento:** Tokenización a nivel de caracteres y estructuración del dataset utilizando una ventana deslizante (*sliding window*) de 100 caracteres con estrategia *Many-to-One*.
2.  **Modelado:** Se implementó una arquitectura modular compuesta por una capa de **Embedding** (para representación densa), una capa recurrente configurable (**Simple RNN**, **GRU** o **LSTM**) y una capa lineal de salida.
3.  **Experimentación:** Se entrenaron las tres variantes bajo condiciones controladas (mismos hiperparámetros y optimizador RMSprop), utilizando la **perplejidad (perplexity)** como métrica principal de evaluación.

**Resultados Clave:**
* **Comparativa de Arquitecturas:** La arquitectura **LSTM** demostró ser la más robusta, alcanzando la menor pérdida y perplejidad en validación, seguida muy de cerca por **GRU**. La **Simple RNN** presentó dificultades significativas para capturar dependencias a largo plazo, evidenciando el problema del desvanecimiento del gradiente.
* **Generación de Texto:** Se evaluaron estrategias de *Greedy Search* y *Beam Search*. Se concluyó que el **Beam Search Estocástico con temperatura controlada ($T \approx 0.8$)** es indispensable para romper los bucles repetitivos inherentes a los modelos recurrentes básicos, logrando generar secuencias con mayor naturalidad y respeto por el estilo del autor original.

____

 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Neo820712/NLP_ErlinRey/blob/main/ErlinRey_Desafio_3.ipynb)
______

## Desafío 4: traducción inglés-español con atención y embeddings


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-77AC91?style=for-the-badge&logo=seaborn&logoColor=white)


Este trabajo final aborda la tarea de traducción automática neuronal (neural machine translation), construyendo un sistema capaz de traducir oraciones del inglés al español utilizando el dataset anki.

![Descripción de la imagen](images/tp4.jpg)

La arquitectura base fue un modelo seq2seq (encoder-decoder) implementado en pytorch. sin embargo, para superar el "cuello de botella" de información de los modelos recurrentes simples —donde todo el sentido de la frase debe caber en un solo vector—, se tomaron decisiones de diseño inspiradas en cómo procesamos el lenguaje los humanos:

* **Arquitectura bi-lstm + atención:** se implementó un codificador bidireccional para capturar el contexto pasado y futuro de cada palabra. además, se añadió un mecanismo de atención (attention). esto permite al modelo "enfocarse" en las palabras relevantes del texto original al generar cada palabra traducida, rompiendo la limitación de la memoria estática y mejorando drásticamente la gestión de frases largas.
* **Transferencia de conocimiento:** en lugar de entrenar el vocabulario desde cero, se inyectaron embeddings pre-entrenados (glove para inglés y fasttext para español). la intuición aquí es clara: es más fácil enseñar a traducir a un modelo que ya "conoce" la semántica de las palabras que a uno que debe aprender el idioma y la traducción simultáneamente.

**Resultados y estrategias de generación:**

* **Optimización:** mediante un grid search, se ajustaron hiperparámetros como la tasa de aprendizaje y el tamaño de las capas ocultas, logrando una convergencia estable y evitando el sobreajuste.
* **Decodificación:** se contrastó la generación greedy frente a beam search. el análisis cualitativo mostró que el beam search es indispensable para evitar "tartamudeos" o repeticiones robóticas, ya que evalúa múltiples futuros posibles antes de comprometerse con una traducción, resultando en textos mucho más naturales y fluidos.
* **Métrica:** el modelo alcanzó un bleu score de **39.54**, validando la eficacia de combinar recurrencia bidireccional con atención para capturar la complejidad gramatical del español.
____

 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Neo820712/NLP_ErlinRey/blob/main/ErlinRey_Desafio_4.ipynb)
______
