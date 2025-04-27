from sqlalchemy.orm import Session
from ..models.usuariosModel import Usuario
from ..schemas.userScheme import UsuarioCreate
from ..core.security import get_password_hash

class UserService:
    def __init__(self, db: Session):
        self.db = db

    def get_user_by_email(self, email: str):
        """Obtiene un usuario por su correo electrónico."""
        return self.db.query(Usuario).filter(Usuario.correo == email).first()

    def get_user_by_id(self, user_id: int):
        """Obtiene un usuario por su ID."""
        return self.db.query(Usuario).filter(Usuario.id == user_id).first()

    def update_password(self, user_id: int, new_password: str):
        """Actualiza la contraseña del usuario."""
        user = self.db.query(Usuario).filter(Usuario.id == user_id).first()
        if user:
            user.contrasena = get_password_hash(new_password)
            self.db.commit()
            self.db.refresh(user)
