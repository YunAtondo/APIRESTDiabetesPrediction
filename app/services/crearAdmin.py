from sqlalchemy.orm import Session
from datetime import datetime
from ..models.usuariosModel import Usuario
from ..core.security import get_password_hash




def crear_usuario_admin_predeterminado(db: Session):
    admin_existente = db.query(Usuario).filter_by(correo="admin@gmail.com").first()
    if not admin_existente:
        admin = Usuario(
            nombre="admin",
            correo="admin@gmail.com",
            contrasena=get_password_hash("admin1234"),
            fecha_creacion=datetime.utcnow(),
            sexo="masculino",
            fecha_nacimiento=datetime(2023, 4, 3),
            rol="admin",
            is_active="ACTIVO"
        )
        db.add(admin)
        db.commit()
        print("Usuario administrador predeterminado creado.")
    else:
        print("Usuario administrador ya existe.")