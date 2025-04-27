from sqlalchemy.orm import Session
from ..models.usuariosModel import Usuario
from ..schemas.userScheme import UsuarioCreate, UsuarioUpdate
from ..core.security import get_password_hash

def crear_usuario(db: Session, usuario: UsuarioCreate):
    """
    Crear un nuevo usuario en la base de datos con contraseña hasheada.
    """
    # Crea un diccionario con los datos del usuario
    usuario_data = usuario.model_dump()
    
    # Hashea la contraseña antes de almacenarla
    usuario_data['contrasena'] = get_password_hash(usuario_data['contrasena'])
    
    # Crea el objeto de usuario con la contraseña hasheada
    db_usuario = Usuario(**usuario_data)
    
    db.add(db_usuario)
    db.commit()
    db.refresh(db_usuario)
    return db_usuario

def get_usuario_by_nombre(db: Session, nombre: str):
    """
    Obtener un usuario por su nombre.
    """
    return db.query(Usuario).filter(Usuario.nombre == nombre).first()

def get_usuario_by_id(db: Session, usuario_id: int):
    """
    Obtener un usuario por su ID.
    """
    return db.query(Usuario).filter(Usuario.id == usuario_id).first()

def get_all_usuarios(db: Session):
    """
    Obtener todos los usuarios.
    """
    return db.query(Usuario).all()

def get_active_usuarios(db: Session):
    """
    Obtener todos los usuarios activos.
    """
    return db.query(Usuario).filter(Usuario.is_active == "ACTIVO").all()

def update_usuario(db: Session, usuario_id: int, usuario_update: UsuarioUpdate):
    """
    Actualiza los datos de un usuario existente (soft update).
    """
    db_usuario = get_usuario_by_id(db, usuario_id)
    if not db_usuario:
        return None

    usuario_data = usuario_update.model_dump(exclude_unset=True)
    
    # Si la contraseña está presente en los datos de actualización, la hasheamos
    if 'contrasena' in usuario_data:
        usuario_data['contrasena'] = get_password_hash(usuario_data['contrasena'])

    for key, value in usuario_data.items():
        setattr(db_usuario, key, value)

    db.commit()
    db.refresh(db_usuario)
    return db_usuario

def soft_delete_usuario(db: Session, usuario_id: int):
    """
    Elimina lógicamente un usuario (soft delete) marcándolo como INACTIVO.
    """
    db_usuario = get_usuario_by_id(db, usuario_id)
    if not db_usuario:
        return None

    db_usuario.is_active = "INACTIVO"
    db.commit()
    db.refresh(db_usuario)
    return db_usuario

def restore_usuario(db: Session, usuario_id: int):
    """
    Restaura un usuario eliminado lógicamente (soft delete) marcándolo como ACTIVO.
    """
    db_usuario = get_usuario_by_id(db, usuario_id)
    if not db_usuario:
        return None

    db_usuario.is_active = "ACTIVO"
    db.commit()
    db.refresh(db_usuario)
    return db_usuario
