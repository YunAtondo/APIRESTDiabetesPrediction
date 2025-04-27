from pydantic import BaseModel, EmailStr
from datetime import datetime

class UsuarioBase(BaseModel):
    nombre: str
    correo: EmailStr
    rol: str
    sexo: str
    fecha_nacimiento: datetime

class UsuarioCreate(UsuarioBase):
    correo: EmailStr
    contrasena: str
    nombre: str
    sexo: str
    fecha_nacimiento: datetime
    fecha_creacion: datetime
    rol: str = "usario"  # Valor por defecto para el rol
    is_active: str = "ACTIVO"  # Valor por defecto para is_active

class Usuario(UsuarioBase):
    id: int
    fecha_creacion: datetime
    is_active: str

    class Config:
        from_attributes = True  # Habilita la compatibilidad con ORM
        
class UsuarioUpdate(BaseModel):
    nombre: str | None = None
    sexo: str | None = None
    fecha_nacimiento: datetime | None = None
    is_active: str | None = "ACTIVO"
    rol: str | None = "usuario"  # Valor por defecto para el rol
    contrasena: str | None = None  # Campo opcional para la contraseña

    class Config:
        from_attributes = True  # Habilita la compatibilidad con ORM