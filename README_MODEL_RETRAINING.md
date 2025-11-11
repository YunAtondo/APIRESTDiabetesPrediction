# 🎯 Sistema de Reentrenamiento y Gestión de Modelos ML

## ✅ ¿Qué se ha implementado?

Se ha creado un **sistema completo de gestión de modelos de Machine Learning** que permite:

1. ✅ **Reentrenar** la red neuronal desde el frontend
2. ✅ **Gestionar versiones** de modelos con timestamps
3. ✅ **Seleccionar** qué modelo usar en producción
4. ✅ **Ver métricas** (Accuracy, F1, Precision, Recall) de cada modelo
5. ✅ **Entrenar** con datos de la BD o datasets CSV personalizados
6. ✅ **Subir** nuevos datasets desde el frontend
7. ✅ **Comparar** diferentes versiones de modelos

---

## 📁 Archivos Creados/Modificados

### Nuevos Archivos:

1. **`app/schemas/modelScheme.py`**
   - Esquemas Pydantic para requests/responses
   - `RetrainingRequest`, `ModelMetrics`, `ModelVersionInfo`, etc.

2. **`app/services/retraining_service.py`**
   - Lógica principal de reentrenamiento
   - Gestión de versiones de modelos
   - Carga de datos desde BD o CSV
   
3. **`app/routes/modelRoute.py`**
   - 6 endpoints REST para gestión de modelos
   - Validación de permisos (solo ADMIN)

4. **`API_MODEL_MANAGEMENT.md`**
   - Documentación completa de la API
   - Ejemplos de requests/responses
   - Guía de uso para frontend

5. **`FRONTEND_EXAMPLES.ts`**
   - Código de ejemplo para Angular
   - Servicio completo con todos los métodos
   - Componente de ejemplo con UI

### Archivos Modificados:

1. **`app/main.py`**
   - Agregado `modelRoute` a los routers

---

## 🚀 Endpoints Disponibles

### 1. POST `/model/retrain` 
Reentrena el modelo con configuración personalizable

**Request:**
```json
{
  "use_database": true,
  "epochs": 190,
  "batch_size": 64,
  "learning_rate": 0.001,
  "hidden_size": 64
}
```

**Response:**
```json
{
  "success": true,
  "message": "Modelo reentrenado exitosamente",
  "version": "v20251111_153045",
  "metrics": {
    "accuracy": 0.95,
    "f1_score": 0.94,
    "precision": 0.93,
    "recall": 0.96,
    "loss": 0.15
  },
  "training_time": 125.5
}
```

### 2. GET `/model/versions`
Lista todas las versiones disponibles con métricas

### 3. POST `/model/activate`
Activa una versión específica en producción

### 4. GET `/model/active`
Obtiene información del modelo actualmente activo

### 5. POST `/model/upload-dataset`
Sube un nuevo dataset CSV para entrenamiento

### 6. DELETE `/model/version/{version}`
Elimina una versión específica (no puede ser la activa)

---

## 📊 Métricas Retornadas

Cada modelo incluye:

- **Accuracy**: Porcentaje de predicciones correctas
- **F1-Score**: Media armónica entre precisión y recall
- **Precision**: Proporción de predicciones positivas correctas
- **Recall**: Proporción de casos positivos identificados
- **Loss**: Pérdida del modelo durante entrenamiento
- **Training Samples**: Número de muestras usadas para entrenar
- **Training Time**: Tiempo que tomó el entrenamiento (en segundos)

---

## 🔄 Flujo de Trabajo

### Opción 1: Reentrenar con datos de la Base de Datos

```typescript
// 1. Verificar datos disponibles en BD
const modelsResponse = await modelService.getModelVersions();

// 2. Reentrenar con datos de BD
const result = await modelService.retrainModel({
  use_database: true,
  epochs: 190
});

// 3. Ver métricas del nuevo modelo
console.log('Accuracy:', result.metrics.accuracy);
console.log('F1-Score:', result.metrics.f1_score);

// 4. Activar el nuevo modelo si las métricas son mejores
if (result.metrics.accuracy > 0.90) {
  await modelService.activateModel(result.version);
}
```

### Opción 2: Reentrenar con Dataset CSV Personalizado

```typescript
// 1. Subir nuevo dataset
await modelService.uploadDataset(file);

// 2. Reentrenar con el dataset subido
const result = await modelService.retrainModel({
  use_database: false,
  dataset_name: 'nuevo_dataset.csv',
  epochs: 200
});

// 3. Activar si es mejor
await modelService.activateModel(result.version);
```

---

## 📂 Estructura de Directorios

El sistema crea automáticamente los siguientes directorios:

```
mlModels/
├── versions/              # Versiones guardadas de modelos
│   ├── v20251111_153045/
│   │   ├── model.pth      # Pesos del modelo
│   │   ├── scaler.pkl     # Scaler entrenado
│   │   └── metadata.json  # Métricas y metadata
│   └── v20251110_120000/
│       ├── model.pth
│       ├── scaler.pkl
│       └── metadata.json
├── datasets/              # Datasets subidos
│   ├── dataset1.csv
│   └── dataset2.csv
├── active_model.json      # Referencia al modelo activo
├── best_diabetes_model.pth  # Modelo activo (copia)
└── scaler.pkl             # Scaler activo (copia)
```

---

## 🔐 Seguridad y Permisos

- ✅ Solo usuarios **ADMIN** pueden:
  - Reentrenar modelos
  - Activar/desactivar versiones
  - Subir datasets
  - Eliminar versiones

- ✅ Todos los usuarios autenticados pueden:
  - Ver lista de modelos
  - Ver métricas
  - Ver modelo activo

---

## 📋 Formato del CSV para Datasets

El archivo CSV debe tener estas columnas:

```csv
HbA1c,AGE,BMI,Gender,CLASS
5.2,45,24.5,M,N
6.5,55,28.3,F,P
7.8,62,32.1,M,Y
5.1,38,22.0,F,Negative
6.8,50,29.5,M,Prediabetes
8.5,65,35.2,F,Diabetes
```

**Columnas requeridas:**
- `HbA1c`: float (nivel de hemoglobina glicosilada)
- `AGE`: int (edad)
- `BMI`: float (índice de masa corporal)
- `Gender`: string ('M' o 'F')
- `CLASS`: string ('N', 'P', 'Y' o 'Negative', 'Prediabetes', 'Diabetes')

---

## 🎨 Integración con Frontend (Angular)

### 1. Crear el servicio:

```bash
ng generate service services/model-management
```

### 2. Copiar el código de `FRONTEND_EXAMPLES.ts`

### 3. Crear el componente:

```bash
ng generate component components/model-management
```

### 4. Agregar a las rutas:

```typescript
const routes: Routes = [
  { 
    path: 'models', 
    component: ModelManagementComponent,
    canActivate: [AdminGuard]  // Solo admins
  }
];
```

### 5. Ejemplo de uso en template:

```html
<div class="model-card" *ngFor="let model of models">
  <h3>{{ model.version }}</h3>
  <p>Accuracy: {{ model.metrics.accuracy | percent }}</p>
  <p>F1-Score: {{ model.metrics.f1_score | number:'1.4-4' }}</p>
  
  <button 
    *ngIf="!model.is_active" 
    (click)="activateModel(model.version)">
    Activar
  </button>
</div>
```

---

## ⚡ Características Avanzadas

### 1. **Versionamiento Automático**
Cada modelo se guarda con timestamp único: `v20251111_153045`

### 2. **Rollback Fácil**
Puedes volver a cualquier versión anterior simplemente activándola

### 3. **Comparación de Modelos**
Compara métricas entre versiones antes de activar

### 4. **Parámetros Personalizables**
Ajusta epochs, batch_size, learning_rate, hidden_size

### 5. **Dos Fuentes de Datos**
- Base de datos (datos reales de usuarios)
- CSV personalizado (para experimentos)

### 6. **Métricas Completas**
No solo accuracy, también F1, precision, recall

---

## 🧪 Ejemplos de Uso

### Ejemplo 1: Entrenar con datos actuales de la BD

```typescript
this.modelService.retrainModel({
  use_database: true,
  epochs: 200
}).subscribe(result => {
  console.log('Nuevo modelo:', result.version);
  console.log('Accuracy:', result.metrics.accuracy);
});
```

### Ejemplo 2: Experimento con dataset personalizado

```typescript
// Primero subir el dataset
this.modelService.uploadDataset(file).subscribe(() => {
  // Luego entrenar con él
  this.modelService.retrainModel({
    use_database: false,
    dataset_name: 'experimento_v1.csv',
    epochs: 300,
    learning_rate: 0.0005
  }).subscribe(result => {
    console.log('Experimento completado');
  });
});
```

### Ejemplo 3: Rollback a versión anterior

```typescript
// Si el nuevo modelo no funciona bien, volver al anterior
this.modelService.activateModel('v20251110_120000').subscribe(() => {
  console.log('Volvimos a la versión anterior');
});
```

---

## 📈 Recomendaciones para el Frontend

### UI sugerida:

1. **Dashboard de Modelos**
   - Card del modelo activo (destacado)
   - Tabla de todas las versiones
   - Gráfico de comparación de métricas

2. **Formulario de Reentrenamiento**
   - Radio buttons: BD vs CSV
   - Sliders para parámetros
   - Preview de configuración
   - Botón "Iniciar Entrenamiento"

3. **Barra de Progreso**
   - Mostrar mientras entrena
   - Estimación de tiempo restante
   - Permitir cancelar (opcional)

4. **Notificaciones**
   - Toast cuando termina el entrenamiento
   - Comparación con modelo anterior
   - Sugerencia de activar si es mejor

---

## ⚠️ Consideraciones Importantes

1. **El reentrenamiento puede tardar**
   - Depende de la cantidad de datos
   - Típicamente 2-5 minutos
   - Mostrar loading/progress en UI

2. **No se puede eliminar el modelo activo**
   - Primero activa otro modelo
   - Luego elimina el antiguo

3. **Mínimo 10 registros en BD**
   - Para entrenar con BD se necesitan al menos 10 registros
   - Mostrar advertencia si no hay suficientes

4. **Validar CSV en frontend**
   - Verificar columnas antes de subir
   - Evitar errores del servidor

5. **Confirmaciones importantes**
   - Confirmar antes de reentrenar
   - Confirmar antes de activar
   - Confirmar antes de eliminar

---

## 🚀 ¡Listo para Usar!

El sistema está completamente funcional. Solo necesitas:

1. ✅ Reiniciar el servidor FastAPI
2. ✅ Implementar el servicio y componente en Angular
3. ✅ Probar con datos de prueba
4. ✅ Configurar permisos de ADMIN en tu app

---

## 📞 Testing de la API

Puedes probar con cURL:

```bash
# Listar modelos
curl -X GET http://localhost:8000/model/versions \
  -H "Authorization: Bearer YOUR_TOKEN"

# Reentrenar
curl -X POST http://localhost:8000/model/retrain \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "use_database": true,
    "epochs": 190
  }'

# Activar modelo
curl -X POST http://localhost:8000/model/activate \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "v20251111_153045"
  }'
```

---

¡Todo listo para producción! 🎉
