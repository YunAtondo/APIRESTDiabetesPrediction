from pydantic import BaseModel
from datetime import datetime

class RecomendacionBase(BaseModel):
    id_usuario: int
    id_registro: int
    recomendacion: str

class RecomendacionCreate(RecomendacionBase):
    pass

class Recomendacion(RecomendacionBase):
    id: int
    fecha_generada: datetime

    class Config:
        from_attributes = True