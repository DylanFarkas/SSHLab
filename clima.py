import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Simulación de datos de ejemplo
data = {
    'temperatura': [30, 25, 20, 18, 15, 10, 12, 25, 22, 20],
    'humedad': [80, 60, 5, 85, 90, 95, 70, 65, 80, 55],
    'viento': [5, 10, 12, 3, 2, 1, 4, 6, 5, 2],
    'lluvia': [1, 1, 1, 0, 1, 1, 0, 0, 1, 0]  # 1 para lluvia, 0 para no lluvia
}

# Crear un DataFrame
df = pd.DataFrame(data)

# Características (X) y etiqueta (y)
X = df[['temperatura', 'humedad', 'viento']]
y = df['lluvia']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)

# Predicción para un nuevo día
nuevo_dia = np.array([[23, 70, 5]])  # Ejemplo: 23°C, 70% humedad, 5 km/h viento
prediccion = modelo.predict(nuevo_dia)

if prediccion[0] == 1:
    print("Se predice que mañana va a llover.")
else:
    print("Se predice que mañana no va a llover.")
