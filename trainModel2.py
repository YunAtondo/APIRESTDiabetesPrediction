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
        
        # Seleccionar features relevantes
        features = ['HbA1c', 'AGE', 'BMI', 'Gender']
        target_class = 'CLASS'
        target_reg = 'HbA1c'
        
        X = df[features].values
        y_class = df[target_class].values
        y_reg = df[target_reg].values.reshape(-1, 1)
        
        # Verificar valores extremos
        print("Valores mínimos:", X.min(axis=0))
        print("Valores máximos:", X.max(axis=0))
        
        # Normalizar datos
        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(X)
        
        scaler_y = StandardScaler()
        y_reg_scaled = scaler_y.fit_transform(y_reg)
        
        # Dividir en train/test
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X, y_class, y_reg_scaled, test_size=0.2, random_state=42, stratify=y_class
        )
        
        return (X_train, X_test, 
                y_class_train, y_class_test,
                y_reg_train, y_reg_test,
                scaler_X, scaler_y)
    
    except Exception as e:
        print(f"Error al cargar datos: {str(e)}")
        raise

class DiabetesDataset(Dataset):
    def __init__(self, features, class_labels, reg_labels):
        self.features = torch.FloatTensor(features)
        self.class_labels = torch.LongTensor(class_labels)
        self.reg_labels = torch.FloatTensor(reg_labels)
    
    def __len__(self):
        return len(self.class_labels)
    
    def __getitem__(self, idx):
        return self.features[idx], (self.class_labels[idx], self.reg_labels[idx])

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.regressor = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        class_output = self.classifier(out)
        reg_output = self.regressor(out)
        return class_output, reg_output

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
        (X_train, X_test,
         y_class_train, y_class_test,
         y_reg_train, y_reg_test,
         scaler_X, scaler_y) = load_data()
    except Exception as e:
        print(f"Error en la carga de datos: {e}")
        return

    # DataLoaders
    train_dataset = DiabetesDataset(X_train, y_class_train, y_reg_train)
    test_dataset = DiabetesDataset(X_test, y_class_test, y_reg_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Modelo
    model = NeuralNetwork(input_size, hidden_size, num_classes)
    criterion_class = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Históricos
    train_class_losses = []
    train_reg_losses = []
    test_accuracies = []
    test_mses = []
    epoch_times = []
    
    # Entrenamiento
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        epoch_class_loss = 0
        epoch_reg_loss = 0
        
        for i, (features, (class_labels, reg_labels)) in enumerate(train_loader):
            optimizer.zero_grad()
            class_output, reg_output = model(features)
            loss_class = criterion_class(class_output, class_labels)
            loss_reg = criterion_reg(reg_output, reg_labels)
            total_loss = loss_class + 0.5 * loss_reg
            total_loss.backward()
            optimizer.step()
            epoch_class_loss += loss_class.item()
            epoch_reg_loss += loss_reg.item()
        
        # Tiempo de época
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # Evaluación
        model.eval()
        correct = 0
        total = 0
        mse_total = 0
        
        with torch.no_grad():
            for features, (class_labels, reg_labels) in test_loader:
                class_output, reg_output = model(features)
                _, predicted = torch.max(class_output.data, 1)
                total += class_labels.size(0)
                correct += (predicted == class_labels).sum().item()
                mse_total += criterion_reg(reg_output, reg_labels).item()
        
        accuracy = 100 * correct / total
        avg_mse = mse_total / len(test_loader)
        
        train_class_losses.append(epoch_class_loss/len(train_loader))
        train_reg_losses.append(epoch_reg_loss/len(train_loader))
        test_accuracies.append(accuracy)
        test_mses.append(avg_mse)
        
        # Mostrar progreso
        if (epoch+1) % 10 == 0:
            avg_epoch_time = np.mean(epoch_times[-10:])
            remaining = (num_epochs - epoch - 1) * avg_epoch_time
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Tiempo época: {epoch_time:.2f}s | Promedio: {avg_epoch_time:.2f}s')
            print(f'Tiempo restante: {timedelta(seconds=int(remaining))}')
            print(f'Accuracy: {accuracy:.2f}% | MSE: {avg_mse:.4f}\n')
    
    # Resultados finales
    total_time = time.time() - start_time
    print(f'\nEntrenamiento completado en {timedelta(seconds=int(total_time))}')
    print(f'Tiempo promedio por época: {np.mean(epoch_times):.2f}s')
    print(f'Mejor precisión: {max(test_accuracies):.2f}%')
    print(f'Mejor MSE: {min(test_mses):.4f}')
    
    # Gráficas
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_class_losses)
    plt.title('Pérdida de Clasificación')
    
    plt.subplot(1, 3, 2)
    plt.plot(test_accuracies)
    plt.title('Precisión en Prueba')
    
    plt.subplot(1, 3, 3)
    plt.plot(epoch_times)
    plt.title('Tiempo por Época (s)')
    
    plt.tight_layout()
    plt.show()
    
    # Guardar modelo
    torch.save(model.state_dict(), 'best_diabetes_model.pth')
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')
    print("Modelo guardado")

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    train_model()