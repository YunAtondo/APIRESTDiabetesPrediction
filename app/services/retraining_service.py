import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import RandomOverSampler
import joblib
import os
import json
from datetime import datetime
from typing import Tuple, Dict
from sqlalchemy.orm import Session
from ..models.registrosModel import Registro
from mlModels.trainModel import NeuralNetwork, DiabetesDataset

# Directorio para guardar modelos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "mlModels", "versions")
ACTIVE_MODEL_FILE = os.path.join(BASE_DIR, "mlModels", "active_model.json")
DATASETS_DIR = os.path.join(BASE_DIR, "mlModels", "datasets")

# Crear directorios si no existen
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)


def load_data_from_database(db: Session, include_original_csv: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga datos desde la base de datos y opcionalmente combina con el CSV original
    
    Args:
        db: Sesión de base de datos
        include_original_csv: Si True, combina datos de BD con DiabetesDataset.csv original
    """
    # 1. Obtener datos de la base de datos
    registros = db.query(Registro).filter(Registro.CLASS.isnot(None)).all()
    
    data = []
    for registro in registros:
        try:
            # Convertir datos
            hba1c = float(registro.HbA1c)
            age = int(registro.AGE)
            bmi = float(registro.BMI)
            gender = 1 if registro.Gender.upper() == 'M' else 0
            
            # Convertir clase
            if registro.CLASS == "Negative":
                clase = 0
            elif registro.CLASS == "Prediabetes":
                clase = 1
            elif registro.CLASS == "Diabetes":
                clase = 2
            else:
                continue  # Ignorar registros con clase desconocida
            
            data.append([hba1c, age, bmi, gender, clase])
        except (ValueError, AttributeError):
            continue  # Ignorar registros con datos inválidos
    
    print(f"📊 Datos de BD: {len(data)} registros")
    
    # 2. Si se solicita, agregar datos del CSV original
    if include_original_csv:
        try:
            original_csv_path = os.path.join(BASE_DIR, "mlModels", "DiabetesDataset.csv")
            
            if os.path.exists(original_csv_path):
                df_original = pd.read_csv(original_csv_path)
                
                # Transformar Gender
                df_original['Gender'] = df_original['Gender'].map({'M': 1, 'F': 0})
                
                # Transformar CLASS
                df_original['CLASS'] = df_original['CLASS'].map({
                    'N': 0, 'P': 1, 'Y': 2,
                    'Negative': 0, 'Prediabetes': 1, 'Diabetes': 2
                })
                
                # Filtrar y seleccionar solo las columnas necesarias
                df_original = df_original.dropna(subset=['HbA1c', 'AGE', 'BMI', 'Gender', 'CLASS'])
                df_original = df_original[df_original['CLASS'].isin([0, 1, 2])]
                
                # Agregar al conjunto de datos
                for _, row in df_original.iterrows():
                    data.append([
                        float(row['HbA1c']),
                        int(row['AGE']),
                        float(row['BMI']),
                        int(row['Gender']),
                        int(row['CLASS'])
                    ])
                
                print(f"📊 Datos del CSV original: {len(df_original)} registros")
                print(f"📊 Total combinado: {len(data)} registros")
            else:
                print(f"⚠️  CSV original no encontrado en {original_csv_path}")
        except Exception as e:
            print(f"⚠️  Error al cargar CSV original: {str(e)}")
            print("   Continuando solo con datos de la BD...")
    
    # 3. Validar que haya suficientes datos
    if len(data) < 10:
        raise ValueError(f"No hay suficientes datos para entrenar. Solo hay {len(data)} registros.")
    
    # 4. Convertir a numpy arrays
    df = pd.DataFrame(data, columns=['HbA1c', 'AGE', 'BMI', 'Gender', 'CLASS'])
    
    X = df[['HbA1c', 'AGE', 'BMI', 'Gender']].values
    y = df['CLASS'].values
    
    return X, y


def load_data_from_csv(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Carga datos desde un archivo CSV"""
    filepath = os.path.join(DATASETS_DIR, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"El archivo {filename} no existe en {DATASETS_DIR}")
    
    df = pd.read_csv(filepath)
    
    # Verificar columnas necesarias
    required_columns = ['HbA1c', 'AGE', 'BMI', 'Gender', 'CLASS']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"El CSV debe contener las columnas: {required_columns}")
    
    # Transformar Gender
    df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})
    
    # Transformar CLASS
    df['CLASS'] = df['CLASS'].map({'N': 0, 'P': 1, 'Y': 2, 'Negative': 0, 'Prediabetes': 1, 'Diabetes': 2})
    
    # Eliminar valores nulos
    df = df.dropna()
    df = df[df['CLASS'].isin([0, 1, 2])]
    
    X = df[['HbA1c', 'AGE', 'BMI', 'Gender']].values
    y = df['CLASS'].values
    
    return X, y


def prepare_data(X: np.ndarray, y: np.ndarray) -> Tuple:
    """Prepara los datos para entrenamiento"""
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Sobremuestreo para balancear clases
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_scaled, y)
    
    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    
    # Calcular pesos para la pérdida
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    
    return X_train, X_test, y_train, y_test, scaler, class_weights_tensor


def train_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_weights_tensor: torch.Tensor,
    epochs: int = 190,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    hidden_size: int = 64
) -> Tuple[NeuralNetwork, Dict[str, float]]:
    """Entrena el modelo y retorna métricas"""
    
    input_size = 4
    num_classes = 3
    
    # Crear datasets
    train_dataset = DiabetesDataset(X_train, y_train)
    test_dataset = DiabetesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Crear modelo
    model = NeuralNetwork(input_size, hidden_size, num_classes)
    
    # Configurar entrenamiento
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_f1 = 0.0
    final_metrics = {}
    
    # Entrenamiento
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Evaluación
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calcular métricas
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        
        if f1 > best_f1:
            best_f1 = f1
            final_metrics = {
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'precision': float(precision),
                'recall': float(recall),
                'loss': float(avg_loss)
            }
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, '
                  f'Accuracy: {accuracy:.2%}, F1: {f1:.4f}')
    
    return model, final_metrics


def save_model_version(
    model: NeuralNetwork,
    scaler: StandardScaler,
    metrics: Dict[str, float],
    training_samples: int
) -> str:
    """Guarda una nueva versión del modelo"""
    
    # Generar nombre de versión con timestamp
    version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    version_dir = os.path.join(MODELS_DIR, version)
    os.makedirs(version_dir, exist_ok=True)
    
    # Guardar modelo
    model_path = os.path.join(version_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    
    # Guardar scaler
    scaler_path = os.path.join(version_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    # Guardar metadata
    metadata = {
        'version': version,
        'created_at': datetime.now().isoformat(),
        'metrics': metrics,
        'training_samples': training_samples,
        'is_active': False
    }
    
    metadata_path = os.path.join(version_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return version


def get_all_model_versions() -> list:
    """Obtiene todas las versiones de modelos disponibles"""
    versions = []
    
    if not os.path.exists(MODELS_DIR):
        return versions
    
    for version_name in os.listdir(MODELS_DIR):
        version_dir = os.path.join(MODELS_DIR, version_name)
        metadata_path = os.path.join(version_dir, "metadata.json")
        
        if os.path.isdir(version_dir) and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                versions.append(metadata)
    
    # Ordenar por fecha de creación (más reciente primero)
    versions.sort(key=lambda x: x['created_at'], reverse=True)
    
    return versions


def get_active_model_version() -> str:
    """Obtiene la versión del modelo activo"""
    if os.path.exists(ACTIVE_MODEL_FILE):
        with open(ACTIVE_MODEL_FILE, 'r') as f:
            data = json.load(f)
            return data.get('active_version', 'best_diabetes_model')
    return 'best_diabetes_model'  # Modelo por defecto


def set_active_model_version(version: str) -> bool:
    """Establece una versión como el modelo activo"""
    version_dir = os.path.join(MODELS_DIR, version)
    
    if not os.path.exists(version_dir):
        return False
    
    # Actualizar metadata de versiones
    versions = get_all_model_versions()
    for v in versions:
        metadata_path = os.path.join(MODELS_DIR, v['version'], "metadata.json")
        v['is_active'] = (v['version'] == version)
        with open(metadata_path, 'w') as f:
            json.dump(v, f, indent=2)
    
    # Guardar versión activa
    active_data = {
        'active_version': version,
        'updated_at': datetime.now().isoformat()
    }
    with open(ACTIVE_MODEL_FILE, 'w') as f:
        json.dump(active_data, f, indent=2)
    
    # Copiar archivos al directorio principal de mlModels
    model_src = os.path.join(version_dir, "model.pth")
    scaler_src = os.path.join(version_dir, "scaler.pkl")
    
    model_dst = os.path.join(BASE_DIR, "mlModels", "best_diabetes_model.pth")
    scaler_dst = os.path.join(BASE_DIR, "mlModels", "scaler.pkl")
    
    import shutil
    shutil.copy2(model_src, model_dst)
    shutil.copy2(scaler_src, scaler_dst)
    
    return True


def retrain_model_service(
    db: Session,
    use_database: bool = True,
    dataset_name: str = None,
    epochs: int = 190,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    hidden_size: int = 64
) -> Dict:
    """
    Servicio principal de reentrenamiento
    
    Si use_database=True: Combina datos de BD + CSV original (DiabetesDataset.csv)
    Si use_database=False: Usa solo el CSV especificado en dataset_name
    """
    
    start_time = datetime.now()
    
    try:
        # 1. Cargar datos
        if use_database:
            print("🔄 Cargando datos desde BD + CSV original...")
            X, y = load_data_from_database(db, include_original_csv=True)
        else:
            if not dataset_name:
                raise ValueError("Debe proporcionar un nombre de dataset")
            print(f"🔄 Cargando datos desde {dataset_name}...")
            X, y = load_data_from_csv(dataset_name)
        
        training_samples = len(y)
        print(f"✅ Total de datos cargados: {training_samples} muestras")
        
        # 2. Preparar datos
        print("⚙️  Preparando datos...")
        X_train, X_test, y_train, y_test, scaler, class_weights_tensor = prepare_data(X, y)
        
        # 3. Entrenar modelo
        print("🚀 Entrenando modelo...")
        model, metrics = train_model(
            X_train, X_test, y_train, y_test, class_weights_tensor,
            epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate, hidden_size=hidden_size
        )
        
        # 4. Guardar versión
        print("💾 Guardando modelo...")
        version = save_model_version(model, scaler, metrics, training_samples)
        
        # Calcular tiempo de entrenamiento
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        print(f"✅ Modelo entrenado exitosamente: {version}")
        print(f"📊 Accuracy: {metrics['accuracy']:.2%}")
        print(f"📊 F1-Score: {metrics['f1_score']:.4f}")
        
        return {
            'success': True,
            'message': f'Modelo reentrenado exitosamente. Nueva versión: {version}',
            'version': version,
            'metrics': metrics,
            'training_time': training_time
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            'success': False,
            'message': f'Error durante el reentrenamiento: {str(e)}',
            'version': None,
            'metrics': None,
            'training_time': 0.0
        }
