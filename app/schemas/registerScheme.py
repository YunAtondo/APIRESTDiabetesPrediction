from pydantic import BaseModel
from datetime import datetime

class RegistroBase(BaseModel):
    id_usuario: int
    AGE: int
    Gender: str
    BMI: str
    HbA1c: str
    CLASS: str

class RegistroCreate(RegistroBase):
    pass

class Registro(RegistroBase):
    id: int
    fecha_registro: datetime
    prediccion_correcta: bool

    class Config:
        from_attributes = True