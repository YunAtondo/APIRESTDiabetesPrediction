# app/services/password_reset_service.py
from fastapi import HTTPException
from sqlalchemy.orm import Session
from typing import Optional

from ..models.usuariosModel import Usuario
from ..schemas.emailScheme import PasswordResetRequest, PasswordReset
from ..services.emailService import EmailService
from ..services.userEmailService import UserService

class PasswordResetService:
    def __init__(self, db: Session, email_service: EmailService):
        self.db = db
        self.email_service = email_service
        self.user_service = UserService(db)

    def request_password_reset(self, request_data: PasswordResetRequest) -> dict:
        """Maneja la solicitud de reseteo de contraseña"""
        user = self.user_service.get_user_by_email(request_data.email)
        
        if not user:
            # Por seguridad, no revelamos si el email existe o no
            return {"message": "Si el email existe, recibirás un enlace para restablecer tu contraseña"}
        
        # Enviar correo con token
        print(f"Llamando a send_reset_email con email: {request_data.email}")# Agrega esta línea
        self.email_service.send_reset_email(request_data.email)
        
        return {"message": "Si el email existe, recibirás un enlace para restablecer tu contraseña"}

    def reset_password(self, reset_data: PasswordReset) -> dict:
        """Maneja el reseteo de contraseña con token válido"""
        # Verificar token
        email = self.email_service.verify_token(reset_data.token)
        
        # Actualizar contraseña
        user = self.user_service.get_user_by_email(email)
        
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        if reset_data.new_password != reset_data.confirm_password:
            raise HTTPException(status_code=400, detail="Las contraseñas no coinciden")
        
        self.user_service.update_password(user.id, reset_data.new_password)
        
        return {"message": "Contraseña actualizada correctamente"}