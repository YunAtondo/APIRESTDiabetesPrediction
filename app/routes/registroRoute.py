from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..models.registrosModel import Registro
from ..schemas.registerScheme import RegistroCreate, Registro as RegistroSchema
from ..database.database import get_db

router = APIRouter(
    prefix="/registros",
    tags=["Registros"]
)

@router.post("/", response_model=RegistroSchema)
def crear_registro(registro: RegistroCreate, db: Session = Depends(get_db)):
    # Lógica similar a usuario_routes
    pass