# 🚂 Deploy en Railway - Opción RECOMENDADA para ML

Railway es **MUCHO MEJOR que Vercel** para aplicaciones con PyTorch porque:
- ✅ Sin límite de tamaño de deployment
- ✅ Sin límite de tiempo de ejecución
- ✅ Base de datos MySQL incluida
- ✅ Mejor para cargas pesadas (ML)
- ✅ $5/mes con $5 gratis iniciales

---

## 🚀 PASOS PARA DEPLOY EN RAILWAY

### 1️⃣ Crear Cuenta en Railway

1. Ve a [railway.app](https://railway.app)
2. Regístrate con GitHub
3. Verifica tu email
4. Añade método de pago (te dan $5 gratis)

### 2️⃣ Crear Nuevo Proyecto

#### Opción A: Desde la Web (Recomendado)

1. Click en "New Project"
2. Selecciona "Deploy from GitHub repo"
3. Autoriza Railway para acceder a tus repos
4. Selecciona tu repositorio `APIRESTDiabetesPrediction`
5. Railway detectará automáticamente que es Python

#### Opción B: Desde CLI

```bash
# Instalar Railway CLI
npm install -g @railway/cli

# O con Cargo (Rust)
cargo install railwayapp

# Login
railway login

# Inicializar proyecto
railway init

# Deploy
railway up
```

### 3️⃣ Añadir Base de Datos MySQL

1. En tu proyecto Railway, click en "+ New"
2. Selecciona "Database"
3. Selecciona "MySQL"
4. Railway creará automáticamente la base de datos

### 4️⃣ Configurar Variables de Entorno

Railway auto-genera algunas variables. Añade las faltantes:

#### Variables AUTO-GENERADAS por Railway:
- `PORT` - Puerto donde corre la app
- `MYSQL_URL` - URL completa de MySQL
- `MYSQL_HOST`
- `MYSQL_PORT`
- `MYSQL_USER`
- `MYSQL_PASSWORD`
- `MYSQL_DATABASE`

#### Variables que DEBES AÑADIR manualmente:

```env
# JWT
SECRET_KEY=tu_clave_secreta_super_segura_aqui
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Email (si usas el servicio de emails)
MAIL_USERNAME=tu_email@gmail.com
MAIL_PASSWORD=tu_app_password_de_gmail
MAIL_FROM=tu_email@gmail.com
MAIL_PORT=587
MAIL_SERVER=smtp.gmail.com
MAIL_FROM_NAME=Diabetes Prediction System

# Base de datos (usar la variable de Railway)
DATABASE_URL=${{MYSQL_URL}}
```

**💡 Nota:** Railway te da `MYSQL_URL` automáticamente, solo usa la variable referenciada.

### 5️⃣ Deploy Automático

Railway hará deploy automáticamente cuando:
- Hagas push a tu rama principal (main/master)
- Detectará `railway.toml` y `Procfile`
- Instalará dependencias de `requirements.txt`
- Ejecutará el comando de `Procfile`

---

## 📋 VERIFICACIÓN POST-DEPLOY

### 1. Check de Logs

En Railway → Tu Proyecto → Deployments → View Logs

Deberías ver:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:XXXX
```

### 2. Probar Endpoints

Railway te da una URL pública como:
```
https://tu-proyecto-production.up.railway.app
```

Prueba:
```bash
# Health check
curl https://tu-proyecto-production.up.railway.app/

# Documentación
https://tu-proyecto-production.up.railway.app/docs
```

### 3. Crear Tablas en MySQL

Railway te da acceso directo a MySQL. Opciones:

#### Opción A: Desde la App (recomendado)
Las tablas se crean automáticamente con SQLAlchemy cuando arranca la app.

#### Opción B: Desde Railway CLI
```bash
railway connect MySQL

# Luego puedes ejecutar SQL:
USE railway;
SHOW TABLES;
```

#### Opción C: Desde un cliente MySQL
Railway te da las credenciales en las variables de entorno.

---

## 🔧 SOLUCIÓN DE PROBLEMAS

### Error: "Application failed to start"

**Revisa logs en Railway:**
```bash
railway logs
```

Causas comunes:
1. Falta variable de entorno `DATABASE_URL`
2. Puerto incorrecto (debe usar `$PORT`)
3. Dependencias no instaladas

**Solución:**
```python
# En app/main.py, asegúrate de usar el puerto de Railway:
import os

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
```

### Error: "MySQL connection refused"

**Verifica que DATABASE_URL esté configurada:**

1. Railway → Variables
2. Añade: `DATABASE_URL=${{MYSQL_URL}}`
3. Redeploy

### Error: "Module not found"

**Dependencias faltantes en requirements.txt**

Revisa que todas las dependencias estén listadas:
```bash
pip freeze > requirements.txt
```

---

## 📊 COMPARACIÓN: Vercel vs Railway

| Feature | Vercel | Railway |
|---------|--------|---------|
| **Precio Free** | Gratis | $5 gratis + $5/mes |
| **Tamaño Max** | 250MB | Sin límite práctico |
| **Tiempo Ejecución** | 10s | Sin límite |
| **Base de Datos** | Externa | MySQL incluido |
| **PyTorch** | ⚠️ Problemático | ✅ Perfecto |
| **Cold Starts** | Lentos | Rápidos |
| **ML Workloads** | ❌ No ideal | ✅ Ideal |

**Recomendación:** Railway para tu proyecto.

---

## 🎯 COMANDOS ÚTILES RAILWAY

```bash
# Ver logs en tiempo real
railway logs

# Abrir app en el navegador
railway open

# Variables de entorno
railway variables

# Conectar a MySQL
railway connect MySQL

# Ver status
railway status

# Rollback a deployment anterior
railway rollback

# Redeploy
railway up --detach
```

---

## 🔐 SEGURIDAD POST-DEPLOY

### 1. CORS para tu Frontend

```python
# En app/main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tu-frontend.vercel.app",  # Tu frontend en producción
        "http://localhost:4200"  # Local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 2. Variables de Entorno Seguras

✅ **NUNCA** hagas commit de `.env`
✅ Usa variables de Railway
✅ Genera SECRET_KEY seguro:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 3. Rate Limiting (Opcional)

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/model/retrain")
@limiter.limit("5/hour")  # Máximo 5 reentrenamientos por hora
async def retrain_model(...):
    ...
```

---

## 📱 CONECTAR FRONTEND ANGULAR

En tu frontend, actualiza la URL del API:

```typescript
// environment.prod.ts
export const environment = {
  production: true,
  apiUrl: 'https://tu-proyecto-production.up.railway.app'
};

// environment.ts (desarrollo)
export const environment = {
  production: false,
  apiUrl: 'http://localhost:8000'
};
```

---

## ✅ CHECKLIST FINAL

Antes de deploy a producción:

- [ ] `railway.toml` configurado
- [ ] `Procfile` creado
- [ ] Variables de entorno configuradas en Railway
- [ ] Base de datos MySQL añadida en Railway
- [ ] CORS configurado para tu frontend
- [ ] `.env` en `.gitignore`
- [ ] `requirements.txt` actualizado
- [ ] SECRET_KEY seguro generado
- [ ] Endpoints probados localmente
- [ ] Push a GitHub
- [ ] Railway hace deploy automático
- [ ] Verificar logs en Railway
- [ ] Probar endpoints en producción
- [ ] Conectar frontend

---

## 🚀 SIGUIENTE PASO

```bash
# 1. Haz commit de los nuevos archivos
git add railway.toml Procfile .vercelignore vercel.json
git commit -m "feat: Add Railway and Vercel deploy configs"
git push origin main

# 2. Ve a railway.app y conecta tu repo

# 3. ¡Listo! Railway hará el deploy automático
```

**¡Tu app de ML estará en producción en 5 minutos! 🎉**
