from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from fastapi import Query

# Importaciones de modelos y schemas
from ..models.recomendacionesPreviasModel import Recomendacion
from ..services.recomendacionesPreviasService import obtener_recomendaciones_por_usuario
from ..schemas.recomendationScheme import RecomendacionOut
from ..database.database import get_db

router = APIRouter(
    prefix="/recomendaciones",
    tags=["Recomendaciones"],
    responses={404: {"description": "No encontrado"}}
)



@router.get("/usuario", response_model=List[RecomendacionOut])
def get_recomendaciones_usuario(
    id_usuario: int = Query(..., description="ID del usuario"),
    db: Session = Depends(get_db)
):
    recomendaciones = obtener_recomendaciones_por_usuario(db, id_usuario)
    return recomendaciones