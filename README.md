# Taller 5: Introducción a las Redes Neuronales Artificiales
**Nombre:** Juan Felipe Pinzón Rincón

## Análisis de los Ejercicios Seleccionados

### 1. Predicción con base en datos experimentales (Regresión con RNA)
* Se utiliza un conjunto de datos experimentales simulados. En este escenario, la red neuronal artificial procesa entradas continuas para generar una predicción numérica (regresión).
* La topología de la red incluye dos capas ocultas. Como se menciona en el documento, la función de activación ReLU es ideal en las capas internas de redes profundas porque mejora la eficiencia del entrenamiento.
* Se dividen los datos en entrenamiento y prueba, aplicando el descenso del gradiente para actualizar gradualmente los pesos a través del algoritmo de retropropagación (backpropagation), reduciendo el error cuadrático medio.

### 2. Clasificación de prendas con Fashion MNIST (Evaluación de Precisión)
* El dataset Fashion MNIST consta de imágenes en escala de grises de $28\times28$ píxeles, similar al MNIST original de dígitos, y busca clasificar cada imagen en una de 10 categorías de prendas de vestir.
* La red se construye con una capa inicial que "aplana" (Flatten) la matriz de $28\times28$ en un vector de 784 características, seguida de dos capas ocultas densas con activación ReLU.
* La capa de salida consta de 10 neuronas con activación `softmax`, la cual produce probabilidades entre 0 y 1 para cada clase. Al final se evalúa la precisión (accuracy) del modelo sobre el conjunto de prueba para validar que no hubo sobre-entrenamiento (overfitting).

### 3. Explicación de funciones principales en Fashion MNIST
* **`Flatten`**: Recibe la imagen bidimensional de los datos de entrada ($28\times28$) y la convierte en un vector unidimensional de 784 píxeles, preparándola para la red neuronal densa.
* **`Dense`**: Representa una capa de neuronas completamente conectadas. Los pesos $W$ y los sesgos $b$ de esta capa se ajustan iterativamente durante el aprendizaje para minimizar la función de pérdida.
* **`ReLU` (Rectified Linear Unit)**: Función de activación $f(x) = \max(0, x)$ que introduce no linealidad y complejidad a la red, resolviendo ineficiencias del entrenamiento presentes en funciones como la sigmoide.
* **`Softmax`**: Función aplicada en la capa de salida que asegura que la suma de las predicciones de las 10 clases sea igual a 1, entregando una distribución de probabilidad clara para la clasificación final.
