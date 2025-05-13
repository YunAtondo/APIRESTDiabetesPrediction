import torch
import joblib
from mlModels.trainModel import NeuralNetwork  # Asegúrate de que el modelo esté disponible

def load_model_and_scaler(model_path: str, scaler_path: str):
    # Primero, creamos el modelo
    input_size = 4  # Número de características (HbA1c, AGE, BMI, Gender)
    hidden_size = 64
    num_classes = 3  # N (0), P (1), Y (2)
    
    model = NeuralNetwork(input_size, hidden_size, num_classes)  # Recuerda usar la misma estructura que en el entrenamiento
    
    # Cargar los pesos del modelo
    model.load_state_dict(torch.load(model_path))  # Cargar los pesos entrenados
    model.eval()  # Poner el modelo en modo evaluación

    # Cargar el scaler
    with open(scaler_path, 'rb') as f:
        scaler = joblib.load(scaler_path)  # Cargar el scaler guardado
    
    return model, scaler