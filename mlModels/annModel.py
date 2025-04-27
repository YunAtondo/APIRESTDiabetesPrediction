import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Cargar y preparar datos
def load_data():
    try:
        df = pd.read_csv('DiabetesDataset.csv')
        
        # Verificar y limpiar datos
        print("Valores únicos en CLASS antes de limpieza:", df['CLASS'].unique())
        
        # Transformar variables categóricas con manejo de valores faltantes/inválidos
        df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})
        df['CLASS'] = df['CLASS'].map({'N': 0, 'P': 1, 'Y': 2})
        
        # Eliminar filas con valores faltantes o inválidos
        df = df.dropna()
        df = df[df['CLASS'].isin([0, 1, 2])]  # Solo mantener clases válidas
        
        # Verificar distribución de clases
        print("Distribución de clases:", df['CLASS'].value_counts())
        
        # Seleccionar features relevantes
        features = ['HbA1c', 'AGE', 'BMI', 'Gender']
        target = 'CLASS'
        
        X = df[features].values
        y = df[target].values
        
        # Verificar valores extremos
        print("Valores mínimos:", X.min(axis=0))
        print("Valores máximos:", X.max(axis=0))
        
        # Normalizar datos
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Dividir en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, scaler
    
    except Exception as e:
        print(f"Error al cargar datos: {str(e)}")
        raise

# Dataset personalizado para PyTorch con verificación de datos
class DiabetesDataset(Dataset):
    def __init__(self, features, labels):
        # Convertir a arrays numpy primero
        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Verificar que no hay NaN/inf
        assert not np.any(np.isnan(features)), "Hay valores NaN en features"
        assert not np.any(np.isinf(features)), "Hay valores infinitos en features"
        assert not np.any(np.isnan(labels)), "Hay valores NaN en labels"
        
        # Verificar rangos de las etiquetas
        unique_labels = np.unique(labels)
        assert set(unique_labels).issubset({0, 1, 2}), f"Etiquetas inválidas encontradas: {unique_labels}"
        
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
load_data()