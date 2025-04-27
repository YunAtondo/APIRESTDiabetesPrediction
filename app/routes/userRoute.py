from app.core.security import get_password_hash
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..models.usuariosModel import Usuario
from ..schemas.userScheme import UsuarioCreate, Usuario as UsuarioSchema , UsuarioUpdate
from ..database.database import get_db
from ..services.userService import crear_usuario, get_usuario_by_nombre, get_usuario_by_id, get_active_usuarios

router = APIRouter(
    prefix="/usuarios",  # Prefijo para todas las rutas aquí
    tags=["Usuarios"]    # Etiqueta para Swagger
)

@router.post("/", response_model=UsuarioSchema, status_code=201)
def crear_usuario_endpoint(usuario: UsuarioCreate, db: Session = Depends(get_db)):
    db_usuario = get_usuario_by_nombre(db, nombre=usuario.nombre)
    if db_usuario:
        raise HTTPException(status_code=400, detail="El usuario ya existe")
    return crear_usuario(db=db, usuario=usuario)

@router.get("/{usuario_id}", response_model=UsuarioSchema)
def obtener_usuario(usuario_id: int, db: Session = Depends(get_db)):
    db_usuario = get_usuario_by_id(db, usuario_id=usuario_id)
    if not db_usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return db_usuario

@router.get("/activos", response_model=list[UsuarioSchema])
def obtener_usuarios_activos(db: Session = Depends(get_db)):
    usuarios_activos = db.query(Usuario).filter(Usuario.is_active == "ACTIVO").all()
    return usuarios_activos

@router.get("/", response_model=list[UsuarioSchema])
def obtener_usuarios(db: Session = Depends(get_db)):
    usuarios = db.query(Usuario).all()
    return usuarios


@router.put("/update/{usuario_id}", response_model=UsuarioSchema)
def actualizar_usuario(usuario_id: int, usuario_update: UsuarioUpdate, db: Session = Depends(get_db)):
    db_usuario = get_usuario_by_id(db, usuario_id=usuario_id)
    if not db_usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    # Si la contraseña está presente en los datos de actualización, la hasheamos
    if usuario_update.contrasena:
        usuario_update.contrasena = get_password_hash(usuario_update.contrasena)
    
    # Actualiza los datos del usuario
    for key, value in usuario_update.model_dump(exclude_unset=True).items():
        setattr(db_usuario, key, value)

    db.commit()
    db.refresh(db_usuario)
    return db_usuario

@router.put("/delete/{usuario_id}", response_model=UsuarioSchema)
def eliminar_usuario(usuario_id: int, db: Session = Depends(get_db)):
    db_usuario = get_usuario_by_id(db, usuario_id=usuario_id)
    if not db_usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    # Cambia el estado a inactivo
    db_usuario.is_active = "INACTIVO"
    db.commit()
    db.refresh(db_usuario)
    return db_usuario

@router.put("/restore/{usuario_id}", response_model=UsuarioSchema)
def restaurar_usuario(usuario_id: int, db: Session = Depends(get_db)):
    db_usuario = get_usuario_by_id(db, usuario_id=usuario_id)
    if not db_usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    # Cambia el estado a activo
    db_usuario.is_active = "ACTIVO"
    db.commit()
    db.refresh(db_usuario)
    return db_usuario

