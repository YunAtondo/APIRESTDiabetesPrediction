import torch
import joblib
from mlModels.trainModel import NeuralNetwork  # Asegúrate de que el modelo esté disponible

def load_model_and_scaler(model_path: str, scaler_path: str):
    """
    Carga el modelo de PyTorch y el scaler.
    Optimizado para usar CPU (Render free tier no tiene GPU).
    """
    try:
        # Primero, creamos el modelo
        input_size = 4  # Número de características (HbA1c, AGE, BMI, Gender)
        hidden_size = 64
        num_classes = 3  # N (0), P (1), Y (2)
        
        model = NeuralNetwork(input_size, hidden_size, num_classes)
        
        # Cargar los pesos del modelo (forzar CPU)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Poner el modelo en modo evaluación
        
        # Cargar el scaler
        scaler = joblib.load(scaler_path)
        
        print(f"✅ Modelo cargado desde: {model_path}")
        print(f"✅ Scaler cargado desde: {scaler_path}")
        
        return model, scaler
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        raise Exception(f"No se pudo cargar el modelo: {str(e)}")