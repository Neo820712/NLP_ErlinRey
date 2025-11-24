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


## Desafío 3: Modelado de Lenguaje con Redes Recurrentes (RNN, GRU, LSTM)

Este desafío se centró en la construcción y evaluación comparativa de modelos de lenguaje a nivel de caracteres (*Character-Level Language Models*), utilizando arquitecturas de redes neuronales recurrentes implementadas en **PyTorch**.

El objetivo principal fue entrenar un modelo capaz de aprender la estructura gramatical y el estilo literario de un corpus complejo (*Don Quijote de la Mancha*), para posteriormente generar texto coherente mediante diferentes estrategias de decodificación.

El flujo de trabajo incluyó:
1.  **Preprocesamiento:** Tokenización a nivel de caracteres y estructuración del dataset utilizando una ventana deslizante (*sliding window*) de 100 caracteres con estrategia *Many-to-One*.
2.  **Modelado:** Se implementó una arquitectura modular compuesta por una capa de **Embedding** (para representación densa), una capa recurrente configurable (**Simple RNN**, **GRU** o **LSTM**) y una capa lineal de salida.
3.  **Experimentación:** Se entrenaron las tres variantes bajo condiciones controladas (mismos hiperparámetros y optimizador RMSprop), utilizando la **perplejidad (perplexity)** como métrica principal de evaluación.

**Resultados Clave:**
* **Comparativa de Arquitecturas:** La arquitectura **LSTM** demostró ser la más robusta, alcanzando la menor pérdida y perplejidad en validación, seguida muy de cerca por **GRU**. La **Simple RNN** presentó dificultades significativas para capturar dependencias a largo plazo, evidenciando el problema del desvanecimiento del gradiente.
* **Generación de Texto:** Se evaluaron estrategias de *Greedy Search* y *Beam Search*. Se concluyó que el **Beam Search Estocástico con temperatura controlada ($T \approx 0.8$)** es indispensable para romper los bucles repetitivos inherentes a los modelos recurrentes básicos, logrando generar secuencias con mayor naturalidad y respeto por el estilo del autor original.
