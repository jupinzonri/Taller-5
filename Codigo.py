"""
Taller_5_IA.py
Implementación de Redes Neuronales Artificiales usando TensorFlow y Keras.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# =====================================================================
# EJERCICIO 1: Modelo de predicción con datos experimentales
# =====================================================================
def ejercicio_1_prediccion_experimental():
    print("--- EJERCICIO 1: PREDICCIÓN CON DATOS EXPERIMENTALES ---")
    
    # Simulación de un dataset experimental (Ej: Voltaje vs Temperatura)
    np.random.seed(42)
    X_exp = np.linspace(0, 10, 200).reshape(-1, 1)
    # Función no lineal con ruido
    Y_exp = 3 * np.sin(X_exp) + 0.5 * X_exp + np.random.randn(200, 1) * 0.5 
    
    # Topología: 2 capas ocultas
    modelo_exp = keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=[1]), # Capa Oculta 1
        layers.Dense(16, activation='relu'),                  # Capa Oculta 2
        layers.Dense(1)                                       # Capa de Salida (Lineal)
    ])
    
    # Compilar con error cuadrático medio (MSE)
    modelo_exp.compile(optimizer='adam', loss='mean_squared_error')
    
    # Entrenamiento
    print("Entrenando modelo experimental...")
    historial = modelo_exp.fit(X_exp, Y_exp, epochs=150, verbose=0, validation_split=0.2)
    
    # Predicción con nuevos datos
    X_nuevo = np.array([[2.5], [5.0], [7.5]])
    predicciones = modelo_exp.predict(X_nuevo)
    
    print("Predicciones para nuevos datos:")
    for x_val, y_pred in zip(X_nuevo, predicciones):
        print(f"Entrada X: {x_val[0]:.2f} -> Predicción Y: {y_pred[0]:.2f}")
    
    print(f"Pérdida final (MSE): {historial.history['loss'][-1]:.4f}\n")


# =====================================================================
# EJERCICIO 2 y 3: Clasificación Fashion MNIST con 2 Capas Ocultas
# =====================================================================
def ejercicios_2_y_3_fashion_mnist():
    print("--- EJERCICIOS 2 Y 3: FASHION MNIST ---")
    
    # Descargar el dataset Fashion MNIST
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    # Nombres de las clases para referencia
    class_names = ['Camiseta', 'Pantalón', 'Saco', 'Vestido', 'Abrigo',
                   'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Bota']
    
    # Normalización: Escalar valores de pixeles entre 0 y 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    # Topología de la Red: 2 capas ocultas
    modelo_fashion = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)), # Capa de Entrada: Aplana matriz 28x28 a vector 784
        layers.Dense(128, activation='relu'), # Capa Oculta 1
        layers.Dense(64, activation='relu'),  # Capa Oculta 2
        layers.Dense(10, activation='softmax')# Capa de Salida: 10 clases, probabilidad
    ])
    
    # Compilación: Función de pérdida para categorías (rótulos enteros)
    modelo_fashion.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
    
    # Entrenamiento
    print("Entrenando modelo Fashion MNIST...")
    historial_fashion = modelo_fashion.fit(train_images, train_labels, epochs=10, validation_split=0.1, verbose=1)
    
    # Evaluación con el conjunto de prueba
    test_loss, test_acc = modelo_fashion.evaluate(test_images,  test_labels, verbose=2)
    print(f"\nPrecisión en el conjunto de prueba: {test_acc:.4f}")
    
    # Conclusiones impresas
    print("\n--- Conclusiones ---")
    print("1. La precisión en el conjunto de prueba indica qué tan bien la red neuronal generaliza patrones de ropa (detectores locales) a datos que nunca ha visto.")
    print("2. Si la precisión de entrenamiento es mucho mayor que la de prueba, tendríamos un problema de 'Overfitting' (sobre-entrenamiento).")
    print("3. La función 'softmax' fue clave porque permite que la red entregue una probabilidad definida (Ej: 90% Bota, 10% Sandalia) facilitando la clasificación final.")


# Ejecución de los scripts
if __name__ == "__main__":
    ejercicio_1_prediccion_experimental()
    ejercicios_2_y_3_fashion_mnist()
"""
