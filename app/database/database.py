from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()


SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

# Render a veces usa 'postgres://' pero SQLAlchemy necesita 'postgresql://'
if SQLALCHEMY_DATABASE_URL and SQLALCHEMY_DATABASE_URL.startswith("postgres://"):
    SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace("postgres://", "postgresql://", 1)

print(f"Database URL configured: {SQLALCHEMY_DATABASE_URL[:30]}...")  # Log parcial para debug

# Configuración del engine con SSL y pool para Render PostgreSQL
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={
        "sslmode": "require",  # Requerir SSL
        "connect_timeout": 10   # Timeout de 10 segundos
    },
    pool_pre_ping=True,  # Verificar conexiones antes de usarlas
    pool_size=5,  # Número de conexiones en el pool
    max_overflow=10,  # Conexiones adicionales permitidas
    pool_recycle=3600  # Reciclar conexiones cada hora
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()