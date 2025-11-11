from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from ..database.database import get_db
from ..schemas.modelScheme import (
    RetrainingRequest, RetrainingResponse, ModelListResponse,
    ActivateModelRequest, ModelVersionInfo, ModelMetrics
)
from ..services.retraining_service import (
    retrain_model_service, get_all_model_versions,
    set_active_model_version, get_active_model_version,
    DATASETS_DIR, MODELS_DIR
)
from ..core.security import get_current_user, TokenData
import os
import base64
from datetime import datetime

router = APIRouter(
    prefix="/model",
    tags=["Model Management"]
)


@router.post("/retrain", response_model=RetrainingResponse)
async def retrain_model(
    request: RetrainingRequest,
    db: Session = Depends(get_db),
    current_user: TokenData = Depends(get_current_user)
):
    """
    Reentrena el modelo de diabetes con datos de la base de datos o un dataset CSV.
    
    Solo usuarios con rol ADMIN pueden reentrenar el modelo.
    """
    # Verificar que el usuario sea administrador (case-insensitive)
    if current_user.rol.upper() != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Solo los administradores pueden reentrenar el modelo"
        )
    
    # Ejecutar reentrenamiento
    result = retrain_model_service(
        db=db,
        use_database=request.use_database,
        dataset_name=request.dataset_name,
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        hidden_size=request.hidden_size
    )
    
    if not result['success']:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result['message']
        )
    
    return RetrainingResponse(
        success=result['success'],
        message=result['message'],
        version=result['version'],
        metrics=ModelMetrics(**result['metrics']),
        training_time=result['training_time']
    )


@router.get("/versions", response_model=ModelListResponse)
async def list_model_versions(
    current_user: TokenData = Depends(get_current_user)
):
    """
    Lista todas las versiones de modelos disponibles.
    """
    versions = get_all_model_versions()
    active_version = get_active_model_version()
    
    model_versions = []
    for v in versions:
        model_versions.append(
            ModelVersionInfo(
                version=v['version'],
                created_at=datetime.fromisoformat(v['created_at']),
                metrics=ModelMetrics(**v['metrics']),
                is_active=v.get('is_active', False),
                training_samples=v['training_samples']
            )
        )
    
    return ModelListResponse(
        models=model_versions,
        active_model=active_version
    )


@router.post("/activate")
async def activate_model_version(
    request: ActivateModelRequest,
    current_user: TokenData = Depends(get_current_user)
):
    """
    Activa una versión específica del modelo para usar en producción.
    
    Solo usuarios con rol ADMIN pueden cambiar el modelo activo.
    """
    # Verificar que el usuario sea administrador (case-insensitive)
    if current_user.rol.upper() != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Solo los administradores pueden cambiar el modelo activo"
        )
    
    success = set_active_model_version(request.version)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"La versión {request.version} no existe"
        )
    
    return {
        "success": True,
        "message": f"Modelo {request.version} activado exitosamente",
        "active_version": request.version
    }


@router.post("/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    current_user: TokenData = Depends(get_current_user)
):
    """
    Sube un nuevo dataset CSV para entrenamiento.
    
    El CSV debe contener las columnas: HbA1c, AGE, BMI, Gender, CLASS
    - Gender: M o F
    - CLASS: N (Negative), P (Prediabetes), Y (Diabetes)
    
    Solo usuarios con rol ADMIN pueden subir datasets.
    """
    # Verificar que el usuario sea administrador (case-insensitive)
    if current_user.rol.upper() != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Solo los administradores pueden subir datasets"
        )
    
    # Verificar que sea un archivo CSV
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo debe ser un CSV"
        )
    
    try:
        # Leer contenido del archivo
        content = await file.read()
        
        # Guardar archivo en el directorio de datasets
        filepath = os.path.join(DATASETS_DIR, file.filename)
        with open(filepath, 'wb') as f:
            f.write(content)
        
        # Verificar que el CSV tenga las columnas correctas
        import pandas as pd
        df = pd.read_csv(filepath)
        required_columns = ['HbA1c', 'AGE', 'BMI', 'Gender', 'CLASS']
        
        if not all(col in df.columns for col in required_columns):
            os.remove(filepath)  # Eliminar archivo inválido
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"El CSV debe contener las columnas: {required_columns}"
            )
        
        return {
            "success": True,
            "message": f"Dataset '{file.filename}' subido exitosamente",
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar el archivo: {str(e)}"
        )


@router.get("/active")
async def get_active_model(
    current_user: TokenData = Depends(get_current_user)
):
    """
    Obtiene información del modelo actualmente activo.
    """
    active_version = get_active_model_version()
    versions = get_all_model_versions()
    
    active_info = None
    for v in versions:
        if v['version'] == active_version:
            active_info = v
            break
    
    if not active_info:
        return {
            "version": active_version,
            "message": "Usando modelo por defecto"
        }
    
    return {
        "version": active_version,
        "created_at": active_info['created_at'],
        "metrics": active_info['metrics'],
        "training_samples": active_info['training_samples']
    }


@router.delete("/version/{version}")
async def delete_model_version(
    version: str,
    current_user: TokenData = Depends(get_current_user)
):
    """
    Elimina una versión específica del modelo.
    
    No se puede eliminar el modelo activo.
    Solo usuarios con rol ADMIN pueden eliminar modelos.
    """
    # Verificar que el usuario sea administrador (case-insensitive)
    if current_user.rol.upper() != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Solo los administradores pueden eliminar modelos"
        )
    
    active_version = get_active_model_version()
    
    if version == active_version:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se puede eliminar el modelo activo. Primero activa otro modelo."
        )
    
    version_dir = os.path.join(MODELS_DIR, version)
    
    if not os.path.exists(version_dir):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"La versión {version} no existe"
        )
    
    try:
        import shutil
        shutil.rmtree(version_dir)
        
        return {
            "success": True,
            "message": f"Versión {version} eliminada exitosamente"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al eliminar la versión: {str(e)}"
        )
