from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from ..database.database import Base



class Recomendacion(Base):
    __tablename__ = "recomendaciones_previas"

    id = Column(Integer, primary_key=True, index=True)
    id_usuario = Column(Integer, ForeignKey("usuarios.id"), nullable=False)  # Cambiado a Integer
    id_registro = Column(Integer, ForeignKey("registros.id"), nullable=False)  # Cambiado a Integer
    recomendacion = Column(String(500), nullable=False)  # Cambiado a String(10)
    fecha_generada = Column(DateTime, nullable=False)  # Cambiado a DateTime
    is_active = Column(String(100), default="ACTIVO")  # Cambiado a Integer
    registro = relationship("Registro", back_populates="recomendaciones")  # Relación inversa con Registro
    usuario = relationship("Usuario", back_populates="recomendaciones")  # Relación inversa con Usuario
    
