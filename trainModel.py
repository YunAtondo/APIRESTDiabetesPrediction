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

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Capa de batch normalization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Regularización
        self.layer2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        return out

def train_model():
    # Configuración
    input_size = 4  # HbA1c, Age, BMI, Gender
    hidden_size = 64
    num_classes = 3  # N, P, Y
    num_epochs = 200  # Reducido para prueba
    learning_rate = 0.001
    batch_size = 64
    
    # Cargar datos con verificación
    try:
        X_train, X_test, y_train, y_test, scaler = load_data()
        
        # Verificación adicional de datos
        print(f"Forma de X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Clases en y_train: {np.unique(y_train, return_counts=True)}")
        
    except Exception as e:
        print(f"Error en la carga de datos: {e}")
        return

    # Crear DataLoaders
    train_dataset = DiabetesDataset(X_train, y_train)
    test_dataset = DiabetesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Inicializar modelo
    model = NeuralNetwork(input_size, hidden_size, num_classes)
    
    # Pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Entrenamiento
    train_losses = []
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for i, (features, labels) in enumerate(train_loader):
            # Verificación rápida de batch
            if torch.any(labels < 0) or torch.any(labels >= num_classes):
                print(f"Batch inválido encontrado en epoch {epoch}: {labels}")
                continue
                
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validación
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in test_loader:
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Guardar el mejor modelo
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_diabetes_model.pth')
            joblib.dump(scaler, 'scaler.pkl')
    
    # Resultados finales
    print(f'Mejor precisión obtenida: {best_accuracy:.2f}%')
    
    # Graficar pérdida
    plt.plot(train_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

if __name__ == '__main__':
    # Configuración de reproducibilidad
    torch.manual_seed(42)
    np.random.seed(42)
    
    train_model()