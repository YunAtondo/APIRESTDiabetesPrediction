from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from ..database.database import get_db
from ..services.prediccion_service import clasificar_y_guardar
from typing import Optional

router = APIRouter(
    prefix="/prediccion",
    tags=["Predicción"],
    )

class InputData(BaseModel):
    id_usuario: Optional[int] = None
    AGE: int
    Gender: str
    BMI: float
    HbA1c: float

@router.post("/clasificar")
def clasificar(data: InputData, db: Session = Depends(get_db)):
    clase, registro_id, recomendaciones = clasificar_y_guardar(data, db)
    return {"clasificacion": clase, "registro_id": registro_id, "recomendaciones": recomendaciones}