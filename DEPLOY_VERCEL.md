# 🚀 Deploy en Vercel - Guía Completa

## ⚠️ PROBLEMA INICIAL
Vercel usa **Python 3.12** por defecto, pero PyTorch 2.0.1 requiere **Python ≤ 3.11**

## ✅ SOLUCIONES IMPLEMENTADAS

### 1️⃣ Archivos Creados

#### `vercel.json` - Configuración de Vercel
```json
{
  "builds": [
    {
      "src": "app/main.py",
      "use": "@vercel/python",
      "config": {
        "runtime": "python3.11"  // ← Fuerza Python 3.11
      }
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app/main.py"
    }
  ]
}
```

#### `runtime.txt` - Especifica versión de Python
```
python-3.11
```

#### `.vercelignore` - Archivos a ignorar en deploy
- Cache de Python
- Modelos entrenados (muy pesados)
- Jupyter notebooks
- Tests

### 2️⃣ Modificación de `requirements.txt`

**Antes:**
```
torch==2.0.1
torchaudio==2.0.2
torchvision==0.15.2
```

**Ahora:**
```
torch>=2.0.1
torchaudio>=2.0.2
torchvision>=0.15.2
```

Esto permite que Vercel instale versiones compatibles con Python 3.11 o 3.12.

---

## 📋 PASOS PARA DEPLOY

### 1. Instalar Vercel CLI (opcional)
```bash
npm install -g vercel
```

### 2. Configurar Variables de Entorno en Vercel

Ve a tu proyecto en Vercel → Settings → Environment Variables:

```env
# Base de datos
DATABASE_URL=mysql+pymysql://USER:PASSWORD@HOST:PORT/DATABASE

# JWT
SECRET_KEY=tu_clave_secreta_aqui
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Email (si usas)
MAIL_USERNAME=tu_email@gmail.com
MAIL_PASSWORD=tu_app_password
MAIL_FROM=tu_email@gmail.com
MAIL_PORT=587
MAIL_SERVER=smtp.gmail.com
MAIL_FROM_NAME=Diabetes App
```

### 3. Deploy desde GitHub

#### Opción A: Desde la Web de Vercel
1. Ve a [vercel.com](https://vercel.com)
2. Click en "Add New Project"
3. Importa tu repositorio de GitHub
4. Vercel detectará automáticamente `vercel.json`
5. Click en "Deploy"

#### Opción B: Desde CLI
```bash
vercel --prod
```

---

## ⚠️ LIMITACIONES DE VERCEL PARA APPS ML

### ❌ Problemas Conocidos:
1. **Límite de tamaño**: 250MB máximo (PyTorch es pesado)
2. **Tiempo de ejecución**: 10 segundos máximo por request
3. **Memoria**: Limitada en el plan gratuito
4. **Cold starts**: Primera request puede tardar 10-20 segundos

### ✅ ALTERNATIVAS RECOMENDADAS

Si tienes problemas con Vercel, usa estas plataformas mejores para ML:

#### 1. **Railway** (Recomendado para ML)
```bash
# Crear railway.toml
```
- ✅ Soporta apps pesadas de ML
- ✅ Base de datos incluida
- ✅ Sin límite de tiempo de ejecución
- ✅ $5/mes con $5 gratis

#### 2. **Render**
- ✅ Plan gratuito con 750 horas/mes
- ✅ Soporta PyTorch
- ✅ Base de datos PostgreSQL gratis

#### 3. **Fly.io**
- ✅ Excelente para FastAPI
- ✅ Soporta GPU (pago)
- ✅ Plan gratuito generoso

---

## 🔧 SOLUCIÓN DE PROBLEMAS

### Error: "torch==2.0.1 not found"
**Ya solucionado** con `torch>=2.0.1` en requirements.txt

### Error: "Deployment exceeds size limit"
PyTorch es muy pesado (~700MB). Soluciones:

1. **Remover dependencias innecesarias:**
```bash
# Crear requirements-vercel.txt solo con lo esencial
fastapi
uvicorn
sqlalchemy
pymysql
pydantic
python-jose
passlib[bcrypt]
python-multipart
python-dotenv
pandas
numpy
scikit-learn
torch  # Vercel descargará versión optimizada
```

2. **Usar modelo pre-cargado desde S3/Cloudflare:**
```python
# En app/utils/model_loader.py
import requests

def load_model_from_cloud():
    url = "https://tu-bucket.s3.amazonaws.com/best_diabetes_model.pth"
    response = requests.get(url)
    # Guardar y cargar
```

### Error: "Function execution timed out"
El reentrenamiento es muy pesado para Vercel.

**Solución:** Desactivar endpoint de reentrenamiento en producción:
```python
# En app/main.py
import os

if os.getenv("VERCEL_ENV") != "production":
    app.include_router(modelRoute)  # Solo en desarrollo
```

---

## 📊 CONFIGURACIÓN ÓPTIMA PARA VERCEL

### `vercel.json` - Configuración Avanzada
```json
{
  "builds": [
    {
      "src": "app/main.py",
      "use": "@vercel/python",
      "config": {
        "runtime": "python3.11",
        "maxLambdaSize": "250mb"
      }
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app/main.py"
    }
  ],
  "regions": ["iad1"],
  "functions": {
    "app/main.py": {
      "memory": 3008,
      "maxDuration": 30
    }
  }
}
```

---

## 🎯 RECOMENDACIÓN FINAL

**Para este proyecto (ML + PyTorch):**

1. ✅ **Railway** (MEJOR OPCIÓN)
   - Perfecto para FastAPI + PyTorch
   - Sin límites de tamaño
   - Base de datos incluida

2. ⚠️ **Vercel** (Solo si usas modelo ligero)
   - Bueno para APIs simples
   - Requiere optimizaciones

3. ✅ **Render** (Opción intermedia)
   - Buen balance precio/rendimiento

---

## 🚂 BONUS: Deploy en Railway (Recomendado)

### 1. Crear `railway.toml`
```toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "uvicorn app.main:app --host 0.0.0.0 --port $PORT"
```

### 2. Crear `Procfile`
```
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### 3. Deploy
```bash
# Instalar Railway CLI
npm install -g @railway/cli

# Login
railway login

# Deploy
railway up
```

**Variables de entorno en Railway:**
- DATABASE_URL (Railway auto-genera si añades MySQL)
- SECRET_KEY
- ALGORITHM
- Etc.

---

## 📞 AYUDA

Si sigues teniendo problemas, prueba:
1. Usar Railway en lugar de Vercel
2. Reducir el tamaño de requirements.txt
3. Usar modelo pre-entrenado desde la nube
4. Desactivar endpoints pesados en producción
