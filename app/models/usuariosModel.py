from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from ..database.database import Base

class Usuario(Base):
    __tablename__ = "usuarios"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(50), nullable=False)
    correo = Column(String(100), unique=True, nullable=False)
    contrasena = Column(String(100), nullable=False)
    fecha_creacion = Column(DateTime, nullable=False)  # Cambiado a DateTime
    sexo = Column(String(10), nullable=False)  # Cambiado a String(10)
    fecha_nacimiento = Column(DateTime, nullable=False)  # Cambiado a DateTime
    rol = Column(String(20), nullable=False)  # Cambiado a String(20)
    is_active = Column(String(100), default="ACTIVO")  # Cambiado a Integer
     # Relación con Registro
    registros = relationship("Registro", back_populates="usuario", cascade="all, delete-orphan")

    # Relación con Recomendacion
    recomendaciones = relationship("Recomendacion", back_populates="usuario", cascade="all, delete-orphan")