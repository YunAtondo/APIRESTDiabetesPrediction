from pydantic import BaseModel
from datetime import datetime
from ..schemas.registerScheme import Registro
from typing import Optional

class RecomendacionOut(BaseModel):
    id: int
    id_usuario: int
    id_registro: int
    recomendacion: str
    fecha_generada: datetime
    is_active: str
    registro: Optional[Registro]


    class Config:
        from_attributes = True  # Habilita la compatibilidad con ORM