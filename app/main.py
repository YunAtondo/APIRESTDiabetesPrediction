from typing import Union
from sqlalchemy.orm import Session
from fastapi import FastAPI
from .database.database import SessionLocal, engine, Base
from .models.usuariosModel import Usuario 
from .models.registrosModel import Registro 
from .models.recomendacionesPreviasModel import Recomendacion
from .routes import userRoute, registroRoute, recomendacionRoute, authRoute, passwordResetRoute, prediccionRoute, modelRoute
from fastapi.middleware.cors import CORSMiddleware
from .services.crearAdmin import crear_usuario_admin_predeterminado



app = FastAPI(
    title="Diabetes Prediction API",
    description="API para predicción de diabetes con ML",
    version="1.0.0"
)

# Configurar CORS - Permitir todos los orígenes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes
    allow_credentials=False,  # Debe ser False cuando allow_origins=["*"]
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Permitir todos los headers
)

# Crear todas las tablas automáticamente
# Esto creará las tablas si no existen (compatible con SkySQL/MariaDB)
Base.metadata.create_all(bind=engine)

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
app.include_router(modelRoute.router)

with SessionLocal() as db:
    crear_usuario_admin_predeterminado(db)

@app.get("/")
def read_root():
    return {
        "message": "Diabetes Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health_check():
    """Endpoint de salud para Render"""
    return {"status": "healthy", "service": "diabetes-prediction-api"}
