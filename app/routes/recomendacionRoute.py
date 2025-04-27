from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

# Importaciones de modelos y schemas
from ..models.recomendacionesPreviasModel import Recomendacion
from ..models.usuariosModel import Usuario
from ..models.registrosModel import Registro
from ..schemas.recomendationScheme import (
    RecomendacionCreate, 
    Recomendacion as RecomendacionSchema
)
from ..database.database import get_db

router = APIRouter(
    prefix="/recomendaciones",
    tags=["Recomendaciones"],
    responses={404: {"description": "No encontrado"}}
)

# ---- Endpoints ----

@router.post("/", 
          response_model=RecomendacionSchema,
          status_code=status.HTTP_201_CREATED)
def crear_recomendacion(
    recomendacion: RecomendacionCreate, 
    db: Session = Depends(get_db)
):
    """
    Crea una nueva recomendación.
    - Verifica que existan el usuario y registro asociados.
    """
    # Validar que el usuario y registro existan
    usuario = db.query(Usuario).filter(Usuario.id == recomendacion.id_usuario).first()
    registro = db.query(Registro).filter(Registro.id == recomendacion.id_registro).first()
    
    if not usuario or not registro:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuario o Registro no encontrado"
        )
    
    db_recomendacion = Recomendacion(**recomendacion.model_dump())
    db.add(db_recomendacion)
    db.commit()
    db.refresh(db_recomendacion)
    return db_recomendacion



@router.get("/usuario/{usuario_id}", 
         response_model=List[RecomendacionSchema])
def listar_recomendaciones_por_usuario(
    usuario_id: int, 
    db: Session = Depends(get_db)
):
    """
    Lista todas las recomendaciones de un usuario.
    """
    return db.query(Recomendacion)\
        .filter(Recomendacion.id_usuario == usuario_id)\
        .all()