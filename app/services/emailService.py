# app/email_service.py
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from itsdangerous import URLSafeTimedSerializer
from fastapi import HTTPException
from dotenv import load_dotenv

load_dotenv()

class EmailService:
    def __init__(self):
        self.serializer = URLSafeTimedSerializer(os.getenv("JWT_SECRET_KEY"))
        
    def send_reset_email(self, email: str):
        # Crear token con expiración (1 hora)
        print(f"send_reset_email llamado con email: {email}")  # Agrega esta línea
        token = self.serializer.dumps(email, salt='password-reset-salt')
        
        reset_url = f"{os.getenv('FRONTEND_URL')}/changePassword?token={token}"
        
        # Configurar mensaje de correo
        msg = MIMEMultipart()
        msg['From'] = os.getenv("EMAIL_FROM")
        msg['To'] = email
        msg['Subject'] = "Restablecimiento de contraseña"
        
        html = f"""
        <html>
          <body>
            <p>Hola,</p>
            <p>Hemos recibido una solicitud para restablecer tu contraseña.</p>
            <p>Por favor haz clic en el siguiente enlace para continuar:</p>
            <p><a href="{reset_url}">Restablecer contraseña</a></p>
            <p>Si no solicitaste este cambio, ignora este correo.</p>
            <p>El enlace expirará en 1 hora.</p>
          </body>
        </html>
        """
        
        msg.attach(MIMEText(html, 'html'))
        
        # Enviar correo
        try:
            print("Iniciando conexión SMTP...")
            with smtplib.SMTP(os.getenv("SMTP_SERVER"), int(os.getenv("SMTP_PORT"))) as server:
                server.set_debuglevel(1)  # Habilita los registros de depuración
                server.starttls()
                server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD"))
                server.send_message(msg)
            return True
        except Exception as e:
            print(f"Error enviando correo: {e}")  # Agrega esta línea para depurar
            raise HTTPException(status_code=500, detail=f"Error enviando correo: {str(e)}")
    
    def verify_token(self, token: str, max_age: int = 3600):
        try:
            email = self.serializer.loads(
                token,
                salt='password-reset-salt',
                max_age=max_age
            )
            return email
        except Exception as e:
            raise HTTPException(status_code=400, detail="Token inválido o expirado")