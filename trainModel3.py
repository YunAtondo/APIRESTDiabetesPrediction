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
import time
from datetime import timedelta

# Cargar y preparar datos
def load_data():
    try:
        df = pd.read_csv('DiabetesDataset.csv')
        
        # Verificar y limpiar datos
        print("Valores únicos en CLASS antes de limpieza:", df['CLASS'].unique())
        
        # Transformar variables categóricas
        df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})
        df['CLASS'] = df['CLASS'].map({'N': 0, 'P': 1, 'Y': 2})
        
        # Eliminar filas con valores faltantes o inválidos
        df = df.dropna()
        df = df[df['CLASS'].isin([0, 1, 2])]
        
        # Verificar distribución de clases
        print("Distribución de clases:", df['CLASS'].value_counts())
        
        # Calcular promedios de HbA1c por clase para el MSE
        global class_hba1c_means
        class_hba1c_means = df.groupby('CLASS')['HbA1c'].mean().values
        
        # Seleccionar features
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
        
        return X_train, X_test, y_train, y_test, scaler, df
    
    except Exception as e:
        print(f"Error al cargar datos: {str(e)}")
        raise

class DiabetesDataset(Dataset):
    def __init__(self, features, labels):
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
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        return self.classifier(out)

def train_model():
    # Configuración
    input_size = 4
    hidden_size = 64
    num_classes = 3
    num_epochs = 200
    learning_rate = 0.001
    batch_size = 64
    
    # Cargar datos
    try:
        X_train, X_test, y_train, y_test, scaler, df = load_data()
        print(f"Forma de X_train: {X_train.shape}")
        print(f"Clases en y_train: {np.unique(y_train, return_counts=True)}")
    except Exception as e:
        print(f"Error en la carga de datos: {e}")
        return

    # DataLoaders
    train_dataset = DiabetesDataset(X_train, y_train)
    test_dataset = DiabetesDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Modelo
    model = NeuralNetwork(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Históricos
    train_losses = []
    test_accuracies = []
    test_mses = []
    epoch_times = []
    
    # Entrenamiento
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0
        
        for i, (features, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Tiempo por época
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        
        # Evaluación
        model.eval()
        correct = 0
        total = 0
        mse_total = 0
        
        with torch.no_grad():
            for features, labels in test_loader:
                outputs = model(features)
                
                # Clasificación
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # MSE vs clase real
                hba1c_values = features[:, 0]  # Columna de HbA1c
                expected = torch.tensor([class_hba1c_means[label] for label in labels])
                mse_total += torch.mean((hba1c_values - expected)**2).item()
        
        accuracy = 100 * correct / total
        avg_mse = mse_total / len(test_loader)
        
        train_losses.append(epoch_loss/len(train_loader))
        test_accuracies.append(accuracy)
        test_mses.append(avg_mse)
        
        if (epoch+1) % 10 == 0:
            avg_time = np.mean(epoch_times[-10:])
            remaining = timedelta(seconds=avg_time*(num_epochs-epoch-1))
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Loss: {train_losses[-1]:.4f} | Accuracy: {accuracy:.2f}%')
            print(f'MSE vs clase: {avg_mse:.4f} | Tiempo/época: {epoch_time:.2f}s')
            print(f'Tiempo estimado restante: {remaining}\n')
    
    # Resultados finales
    total_time = timedelta(seconds=sum(epoch_times))
    print(f'\nEntrenamiento completado en {total_time}')
    print(f'Mejor precisión: {max(test_accuracies):.2f}%')
    print(f'Último MSE: {test_mses[-1]:.4f}')
    print(f'Velocidad promedio: {np.mean(epoch_times):.2f}s/época')
    
    # Gráficas
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Pérdida de entrenamiento')
    
    plt.subplot(1, 3, 2)
    plt.plot(test_accuracies)
    plt.title('Precisión en prueba')
    
    plt.subplot(1, 3, 3)
    plt.plot(test_mses)
    plt.title('MSE vs clase real')
    
    plt.tight_layout()
    plt.show()
    
    # Guardar modelo
    torch.save(model.state_dict(), 'diabetes_classifier.pth')
    joblib.dump(scaler, 'scaler.pkl')
    print("Modelo guardado")

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    train_model()