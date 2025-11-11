"""
Script para verificar la conexión a SkySQL y crear la base de datos si no existe
"""
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def test_connection():
    """Prueba la conexión a SkySQL"""
    print("🔗 Probando conexión a SkySQL...")
    print(f"📍 URL (sin contraseña): {DATABASE_URL.replace(DATABASE_URL.split(':')[2].split('@')[0], '****')}")
    
    try:
        # Crear engine
        engine = create_engine(DATABASE_URL)
        
        # Probar conexión
        with engine.connect() as connection:
            result = connection.execute(text("SELECT VERSION()"))
            version = result.fetchone()[0]
            print(f"✅ Conexión exitosa!")
            print(f"📊 Versión de MariaDB: {version}")
            
            # Verificar base de datos actual
            result = connection.execute(text("SELECT DATABASE()"))
            db_name = result.fetchone()[0]
            print(f"💾 Base de datos actual: {db_name}")
            
            # Listar tablas
            result = connection.execute(text("SHOW TABLES"))
            tables = result.fetchall()
            if tables:
                print(f"📋 Tablas existentes:")
                for table in tables:
                    print(f"   - {table[0]}")
            else:
                print("📋 No hay tablas aún (se crearán al iniciar la app)")
            
        return True
        
    except Exception as e:
        print(f"❌ Error de conexión: {str(e)}")
        print("\n💡 Verifica:")
        print("   1. La contraseña en el archivo .env")
        print("   2. Que la base de datos 'tesis_db' exista en SkySQL")
        print("   3. Tu conexión a Internet")
        return False

def create_database_if_not_exists():
    """Intenta crear la base de datos si no existe"""
    print("\n🏗️  Intentando crear la base de datos...")
    
    # Obtener URL sin el nombre de la base de datos
    url_parts = DATABASE_URL.rsplit('/', 1)
    base_url = url_parts[0]
    db_name = url_parts[1].split('?')[0]  # Obtener nombre de BD sin parámetros
    
    try:
        # Conectar sin especificar base de datos
        engine = create_engine(base_url + "/?ssl_ca=&ssl_verify_cert=true")
        
        with engine.connect() as connection:
            # Intentar crear la base de datos
            connection.execute(text(f"CREATE DATABASE IF NOT EXISTS {db_name}"))
            connection.commit()
            print(f"✅ Base de datos '{db_name}' creada o ya existe")
            return True
            
    except Exception as e:
        print(f"⚠️  No se pudo crear la base de datos automáticamente: {str(e)}")
        print("\n💡 Crea la base de datos manualmente con:")
        print(f"   CREATE DATABASE {db_name};")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TEST DE CONEXIÓN A SKYSQL (MariaDB)")
    print("=" * 60)
    
    # Primero intentar crear la base de datos
    create_database_if_not_exists()
    
    # Luego probar la conexión
    print()
    success = test_connection()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ ¡Todo listo! Puedes iniciar tu aplicación con:")
        print("   uvicorn app.main:app --reload")
    else:
        print("❌ Por favor, revisa la configuración antes de continuar")
    print("=" * 60)
