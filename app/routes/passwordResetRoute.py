# app/api/endpoints/password_reset.py
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import Annotated

from ..schemas.emailScheme import PasswordResetRequest, PasswordReset
from ..services.passwordResetService import PasswordResetService
from ..services.emailService import EmailService
from ..database.database import get_db
from ..core.emailDependencies import get_email_service

router = APIRouter(tags=["password-reset"])

def get_password_reset_service(
    db: Annotated[Session, Depends(get_db)],
    email_service: Annotated[EmailService, Depends(get_email_service)]
) -> PasswordResetService:
    return PasswordResetService(db, email_service)

@router.post("/request-reset", summary="Solicitar reseteo de contraseña")
async def request_password_reset(
    request_data: PasswordResetRequest,
    service: Annotated[PasswordResetService, Depends(get_password_reset_service)]
):
    print(f"Solicitud de reseteo recibida para: {request_data.email}")
    return service.request_password_reset(request_data)

@router.post("/reset-password", summary="Restablecer contraseña con token")
async def reset_password(
    reset_data: PasswordReset,
    service: Annotated[PasswordResetService, Depends(get_password_reset_service)]
):
    return service.reset_password(reset_data)