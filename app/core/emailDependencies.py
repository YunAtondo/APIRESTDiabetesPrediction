from ..services.emailService import EmailService
from sqlalchemy.orm import Session
from ..database.database import get_db
from ..services.passwordResetService import PasswordResetService

def get_email_service() -> EmailService:
    print("Inicializando EmailService")  # Agrega esta línea
    return EmailService()

def get_passwordResetService(
    db: Session,
    email_service: EmailService
) -> PasswordResetService: 
    print("Inicializando PasswordResetService")
    return PasswordResetService(db, email_service)