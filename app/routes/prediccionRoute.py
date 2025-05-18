from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from ..database.database import get_db
from ..services.prediccion_service import clasificar_y_guardar
from typing import Optional
from ..models.registrosModel import Registro

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

@router.put("/registros/{registro_id}/validar-prediccion")
def validar_prediccion(registro_id: int, es_correcta: bool, db: Session = Depends(get_db)):
    registro = db.query(Registro).filter(Registro.id == registro_id).first()
    if registro:
        registro.prediccion_correcta = es_correcta
        db.commit()
        return {"mensaje": "Predicción validada correctamente"}
    else:
        return {"error": "Registro no encontrado"}