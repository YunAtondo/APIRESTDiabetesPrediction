from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from ..database.database import Base


class Registro(Base):
    __tablename__ = "registros"

    id = Column(Integer, primary_key=True, index=True)
    id_usuario = Column(Integer, ForeignKey("usuarios.id"), nullable=True)  # Cambiado a Integer
    fecha_registro = Column(DateTime, nullable=False)
    AGE = Column(Integer, nullable=False)  # Cambiado a String(10)
    Gender = Column(String(255), nullable=False)  # Cambiado a String(10)
    BMI = Column(String(255), nullable=False)  # Cambiado a String(10)
    HbA1c = Column(String(255), nullable=False)  # Cambiado a String(10)
    CLASS = Column(String(255), nullable=False)  # Cambiado a String(10)
    prediccion_correcta = Column(Boolean, default=True, nullable=True)  # Cambiado a String(10)
    
    # Relación con Usuario
    usuario = relationship("Usuario", back_populates="registros")

    # Relación con Recomendacion
    recomendaciones = relationship("Recomendacion", back_populates="registro", cascade="all, delete-orphan")
