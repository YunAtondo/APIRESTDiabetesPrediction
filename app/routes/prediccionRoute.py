from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from ..database.database import get_db
from ..services.prediccion_service import clasificar_y_guardar
from typing import Optional
from ..models.registrosModel import Registro
import traceback

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
    try:
        clase, registro_id, recomendaciones = clasificar_y_guardar(data, db)
        return {"clasificacion": clase, "registro_id": registro_id, "recomendaciones": recomendaciones}
    except Exception as e:
        print(f"❌ Error en predicción: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error al realizar la predicción: {str(e)}"
        )

@router.put("/registros/{registro_id}/validar-prediccion")
def validar_prediccion(registro_id: int, es_correcta: bool, db: Session = Depends(get_db)):
    registro = db.query(Registro).filter(Registro.id == registro_id).first()
    if registro:
        registro.prediccion_correcta = es_correcta
        db.commit()
        return {"mensaje": "Predicción validada correctamente"}
    else:
        return {"error": "Registro no encontrado"}