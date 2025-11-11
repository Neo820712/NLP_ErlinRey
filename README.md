# NLP_ErlinRey

## Desafío 1: Vectorización y clasificación con 20 Newsgroups

Este desafío implementa un flujo completo de procesamiento de lenguaje natural, desde la representación de texto hasta la clasificación y el análisis semántico, utilizando el dataset *20 Newsgroups*.

El primer paso consistió en la **vectorización de texto** mediante **tf-idf**. Se realizó un análisis iterativo de la similaridad de documentos (usando similitud coseno), demostrando cómo el filtrado de *stop-words* y el ajuste del hiperparámetro `min_df` (frecuencia mínima de documento) son cruciales para reducir el ruido y mejorar la coherencia semántica de los vectores. Sobre esta representación optimizada, se construyó un clasificador *baseline* de **prototipos (1-nn)**, que sirvió como punto de referencia midiendo la precisión basada puramente en la similitud del vecino más cercano.

La segunda fase se centró en la **clasificación probabilística**. Se entrenaron, compararon y optimizaron modelos **Multinomial Naïve Bayes (MNB)** y **Complement Naïve Bayes (CNB)**. Mediante una búsqueda de hiperparámetros (`alpha` y `min_df`), se buscó maximizar el **f1-score macro**, métrica robusta para este tipo de corpus. El análisis demostró la superioridad de CNB sobre MNB para este dataset, logrando una mejor gestión de las clases con vocabularios solapados. Finalmente, se exploró la **similaridad entre palabras** transponiendo la matriz documento-término; este ejercicio demostró cómo la co-ocurrencia contextual permite agrupar términos semánticamente relacionados (ej. "god" con "jesus", "space" con "nasa"), ilustrando el principio fundamental de la semántica distribucional.

## Desafío 2: Word Embeddings y Análisis Semántico con Gensim

En este desafío se abordó la representación distribuida del lenguaje mediante **word embeddings**, aplicando las metodologías vistas en la Clase 2.  
El objetivo fue construir, visualizar e interpretar representaciones densas de palabras a partir de un corpus temático específico.

El corpus seleccionado correspondió a letras de canciones de *Michael Jackson*, extraídas del dataset *Songs*.  
Tras un proceso de preprocesamiento (tokenización, normalización y eliminación de stop-words), se entrenaron dos modelos con **Gensim Word2Vec**: **CBOW** (Continuous Bag of Words) y **Skip-Gram**, utilizando ventanas de contexto y muestreo negativo.

La evaluación incluyó el análisis de palabras más y menos similares, donde CBOW mostró asociaciones más frecuentes y contextuales, mientras que Skip-Gram capturó relaciones semánticas más específicas y expresivas.  
Posteriormente, se aplicó una reducción de dimensionalidad (*PCA → t-SNE*) para proyectar los embeddings en 2D, identificando agrupamientos temáticos coherentes con la lírica del artista: amor, baile, reflexión y performance.

Finalmente, se compararon ambos modelos, concluyendo que **CBOW** produce representaciones más estables y generalistas, mientras que **Skip-Gram** discrimina mejor entre contextos y capta matices líricos más sutiles.  
El trabajo evidenció cómo los embeddings logran encapsular la semántica de un corpus a través de la distribución contextual de las palabras, profundizando los conceptos de la semántica distribucional explorados en el Desafío 1.
