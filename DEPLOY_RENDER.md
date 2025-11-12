# 🚀 Deploy en Render - Guía Completa

Render soporta PyTorch sin problemas y es **GRATIS** (750 horas/mes).

---

## 📋 PASOS PARA DEPLOY

### 1️⃣ Preparar Repositorio

Archivos necesarios (✅ ya creados):
- ✅ `Procfile` - Comando de inicio
- ✅ `runtime.txt` - Python 3.11
- ✅ `build.sh` - Script de build
- ✅ `requirements.txt` - Dependencias con PyTorch

### 2️⃣ Crear Cuenta en Render

1. Ve a [render.com](https://render.com)
2. Sign up con GitHub
3. Conecta tu cuenta de GitHub

### 3️⃣ Crear Web Service

1. Click en **"New +"** → **"Web Service"**
2. Conecta tu repositorio: `APIRESTDiabetesPrediction`
3. Configuración:

```
Name: diabetes-prediction-api
Region: Oregon (US West) o el más cercano
Branch: main
Runtime: Python 3
Build Command: ./build.sh
Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

4. Plan: **Free** (750 horas/mes gratis)

### 4️⃣ Variables de Entorno

En Render → Environment → Add Environment Variables:

```env
DATABASE_URL=mysql+pymysql://USER:PASSWORD@HOST:PORT/DATABASE
SECRET_KEY=tu_clave_secreta_aqui
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
PYTHON_VERSION=3.11.0
```

**Opcional (Email):**
```env
MAIL_USERNAME=tu_email@gmail.com
MAIL_PASSWORD=tu_app_password
MAIL_FROM=tu_email@gmail.com
MAIL_PORT=587
MAIL_SERVER=smtp.gmail.com
MAIL_FROM_NAME=Diabetes Prediction
```

### 5️⃣ Conectar Base de Datos

#### Opción A: Usar SkySQL (Recomendado)
Ya tienes SkySQL configurado, solo usa el DATABASE_URL que tienes.

#### Opción B: PostgreSQL de Render (Gratis)
1. En Render, click **"New +"** → **"PostgreSQL"**
2. Nombre: `diabetes-db`
3. Plan: **Free**
4. Copiar la URL interna
5. **Modificar código** para usar PostgreSQL en lugar de MySQL:

```python
# En .env o variables de Render
DATABASE_URL=postgresql://user:password@host/dbname
```

**IMPORTANTE:** Si usas PostgreSQL, instala:
```bash
# Añadir a requirements.txt:
psycopg2-binary==2.9.9
```

---

## 🔧 CONFIGURACIÓN AVANZADA

### Aumentar Memoria (Si es necesario)

Si el build falla por memoria:

1. Render → Settings → Environment
2. Añadir variable:
```
PYTHON_MAX_MEMORY=2048
```

### Optimizar Build Time

En `build.sh`:
```bash
# Usar cache de pip
pip install --cache-dir=/tmp/pip-cache -r requirements.txt
```

---

## 🚀 DEPLOY

### Automático (Recomendado)

```bash
# Hacer commit
git add .
git commit -m "feat: Add Render deployment config"
git push origin main

# Render detecta el push y hace deploy automático
```

### Manual desde Render

1. Ve a tu servicio en Render
2. Click en **"Manual Deploy"** → **"Deploy latest commit"**

---

## 🧪 TESTING POST-DEPLOY

Render te da una URL como:
```
https://diabetes-prediction-api.onrender.com
```

Probar endpoints:

```bash
# Health check
curl https://diabetes-prediction-api.onrender.com/

# Documentación interactiva
https://diabetes-prediction-api.onrender.com/docs

# Predicción
curl -X POST "https://diabetes-prediction-api.onrender.com/prediccion/clasificar" \
  -H "Content-Type: application/json" \
  -d '{
    "AGE": 50,
    "Gender": "M",
    "BMI": 28.5,
    "HbA1c": 6.2
  }'
```

---

## ⚙️ CONFIGURACIÓN DE CORS

Para conectar tu frontend en Vercel, actualiza `app/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",  # Desarrollo
        "https://tu-frontend.vercel.app",  # Producción
        "https://diabetes-prediction-api.onrender.com"  # Render
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📊 LIMITACIONES DEL PLAN FREE

| Feature | Free Plan |
|---------|-----------|
| **Horas** | 750 horas/mes |
| **Memoria** | 512 MB |
| **CPU** | Compartido |
| **Sleep** | Sí (después de 15 min inactividad) |
| **Build Time** | Hasta 15 minutos |
| **Cold Start** | 30-60 segundos |

### ⚠️ Cold Starts

El plan gratuito duerme la app después de 15 minutos de inactividad.
Primera request después de dormir tarda ~30-60s.

**Solución:** Usar cron job para mantener activo:
```bash
# Cron-job.org cada 10 minutos
curl https://diabetes-prediction-api.onrender.com/health
```

---

## 🆙 UPGRADE A PLAN PAGADO

Si necesitas mejor rendimiento:

**Starter Plan - $7/mes:**
- ✅ Sin sleep
- ✅ 512 MB RAM
- ✅ Cold starts más rápidos

**Standard Plan - $25/mes:**
- ✅ 2 GB RAM
- ✅ Mejor CPU
- ✅ Ideal para ML

---

## 🔍 LOGS Y DEBUGGING

### Ver Logs en Tiempo Real

En Render → Logs → Ver salida en tiempo real

### Comandos útiles:

Render muestra automáticamente:
```
=== Build Logs ===
Installing dependencies...
✅ torch installed successfully

=== Deploy Logs ===
INFO:     Started server process
INFO:     Application startup complete.
```

---

## 🐛 SOLUCIÓN DE PROBLEMAS

### Error: "Build failed"

**Causa:** Memoria insuficiente o timeout

**Solución 1:** Reducir dependencias innecesarias
```bash
# Remover de requirements.txt lo que no uses:
# - jupyter
# - matplotlib
# - seaborn (si no lo usas)
```

**Solución 2:** Usar imagen Docker personalizada
```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### Error: "Module not found"

**Causa:** Falta dependencia en requirements.txt

**Solución:**
```bash
pip freeze > requirements.txt
git add requirements.txt
git commit -m "fix: Add missing dependencies"
git push
```

### Error: "Database connection failed"

**Causa:** DATABASE_URL incorrecta o firewall

**Solución:**
1. Verificar variable de entorno en Render
2. Si usas SkySQL, agregar IP de Render a whitelist
3. O usar PostgreSQL de Render (más fácil)

### Error: "Timeout durante el build"

**Causa:** PyTorch tarda mucho en instalarse

**Solución:** Usar versión optimizada:
```txt
# En requirements.txt
torch==2.0.1+cpu  # Versión CPU más ligera
```

---

## 🎯 COMPARACIÓN: Render vs Railway vs Vercel

| Feature | Render Free | Railway Free | Vercel |
|---------|-------------|--------------|--------|
| **Soporta PyTorch** | ✅ Sí | ✅ Sí | ⚠️ Limitado |
| **Precio** | Gratis | $5 gratis | Gratis |
| **Límite Tamaño** | Sin límite | 4 GB | 250 MB |
| **Sleep** | Sí (15 min) | No | No |
| **Build Time** | Lento (~10 min) | Rápido (~5 min) | Rápido |
| **Base de Datos** | PostgreSQL gratis | MySQL pagado | Externa |
| **Mejor para** | Proyectos ML | Apps Python | Frontends |

---

## ✅ CHECKLIST ANTES DE DEPLOY

- [x] `Procfile` creado
- [x] `runtime.txt` creado
- [x] `build.sh` creado
- [x] `requirements.txt` actualizado
- [ ] Variables de entorno configuradas en Render
- [ ] Base de datos configurada
- [ ] CORS actualizado para producción
- [ ] Commit y push a GitHub
- [ ] Conectar repositorio en Render
- [ ] Verificar logs de build
- [ ] Probar endpoints en producción

---

## 🚀 SIGUIENTE PASO

```bash
# 1. Hacer commit de archivos de Render
git add Procfile runtime.txt build.sh
git commit -m "feat: Add Render deployment configuration"
git push origin main

# 2. Ve a render.com y crea el Web Service

# 3. Espera 10-15 minutos para el build inicial

# 4. ¡Listo! Tu API con ML está en producción 🎉
```

---

## 📞 RECURSOS

- [Documentación Render](https://render.com/docs)
- [Render Community](https://community.render.com/)
- [Status de Render](https://status.render.com/)

**¡Tu app de ML funcionará perfectamente en Render! 🎉**
