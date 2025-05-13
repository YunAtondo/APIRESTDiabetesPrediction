from typing import Union
from sqlalchemy.orm import Session
from fastapi import FastAPI
from .database.database import SessionLocal, engine
from .models.usuariosModel import Usuario 
from .models.registrosModel import Registro 
from .models.recomendacionesPreviasModel import Recomendacion
from .routes import userRoute, registroRoute, recomendacionRoute, authRoute, passwordResetRoute, prediccionRoute
from fastapi.middleware.cors import CORSMiddleware
from .services.crearAdmin import crear_usuario_admin_predeterminado



app = FastAPI()
# Configurar CORS - este middleware debe ser añadido ANTES de cualquier otra configuración de rutas
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # URL de tu aplicación Angular
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crear tablas (solo en desarrollo)
Usuario.metadata.create_all(bind=engine)
Registro.metadata.create_all(bind=engine)
Recomendacion.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
app.include_router(userRoute.router)
app.include_router(registroRoute.router)
app.include_router(recomendacionRoute.router)
app.include_router(authRoute.router)
app.include_router(passwordResetRoute.router)
app.include_router(prediccionRoute.router)

with SessionLocal() as db:
    crear_usuario_admin_predeterminado(db)

def read_root():
    return {"message": "Welcome to the API"}