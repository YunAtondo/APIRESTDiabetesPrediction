from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class RetrainingRequest(BaseModel):
    """Request para reentrenar el modelo"""
    use_database: bool = True  # True: usar datos de BD, False: usar dataset CSV
    dataset_name: Optional[str] = None  # Nombre del archivo CSV si use_database=False
    epochs: Optional[int] = 190
    batch_size: Optional[int] = 64
    learning_rate: Optional[float] = 0.001
    hidden_size: Optional[int] = 64
    
class ModelMetrics(BaseModel):
    """Métricas del modelo"""
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    loss: float
    
class ModelVersionInfo(BaseModel):
    """Información de una versión del modelo"""
    version: str
    created_at: datetime
    metrics: ModelMetrics
    is_active: bool
    training_samples: int
    
class RetrainingResponse(BaseModel):
    """Respuesta del reentrenamiento"""
    success: bool
    message: str
    version: str
    metrics: ModelMetrics
    training_time: float
    
class ModelListResponse(BaseModel):
    """Lista de modelos disponibles"""
    models: List[ModelVersionInfo]
    active_model: str
    
class ActivateModelRequest(BaseModel):
    """Request para activar un modelo"""
    version: str
    
class UploadDatasetRequest(BaseModel):
    """Request para subir un dataset"""
    filename: str
    data: str  # CSV en formato base64 o texto
