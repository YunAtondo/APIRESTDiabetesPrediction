from pydantic import BaseModel

class LoginSchema(BaseModel):
    correo: str
    contrasena: str
    
