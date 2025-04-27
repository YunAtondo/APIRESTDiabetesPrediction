from datetime import timedelta
from typing import Optional
from sqlalchemy.orm import Session
from ..core.security import verify_password, create_access_token
from ..models.usuariosModel import Usuario
from dotenv import load_dotenv
import os

load_dotenv()

ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 3600))  # Default to 30 minutes

def authenticate_user(db: Session, correo: str, contrasena: str) -> Optional[Usuario]:
    """
    Authenticate a user by username and password.
    """
    user = db.query(Usuario).filter(Usuario.correo == correo).first()
    if not user or not verify_password(contrasena, user.contrasena):
        return None
    
    # Validar si el usuario está activo
    if user.is_active != "ACTIVO":
        return None
    return user

def get_Token(user: Usuario) -> dict:
    """
    Generate an access token for the authenticated user.
    """
    access_token_expires = timedelta(minutes= ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.nombre, "rol": user.rol, "userid": user.id}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}