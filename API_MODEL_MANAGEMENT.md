# API de Gestión de Modelos de Machine Learning

Esta API permite reentrenar, gestionar y versionar modelos de predicción de diabetes.

## 🎯 Endpoints Disponibles

### 1. **POST /model/retrain** - Reentrenar el modelo

Reentrena el modelo con datos de la base de datos o un dataset CSV.

**Requiere:** Autenticación + Rol ADMIN

**Request Body:**
```json
{
  "use_database": true,
  "dataset_name": "DiabetesDataset.csv",
  "epochs": 190,
  "batch_size": 64,
  "learning_rate": 0.001,
  "hidden_size": 64
}
```

**Parámetros:**
- `use_database` (bool): `true` para usar datos de la BD, `false` para usar CSV
- `dataset_name` (string, opcional): Nombre del archivo CSV (requerido si `use_database=false`)
- `epochs` (int, opcional): Número de épocas de entrenamiento (default: 190)
- `batch_size` (int, opcional): Tamaño del batch (default: 64)
- `learning_rate` (float, opcional): Tasa de aprendizaje (default: 0.001)
- `hidden_size` (int, opcional): Tamaño de la capa oculta (default: 64)

**Response:**
```json
{
  "success": true,
  "message": "Modelo reentrenado exitosamente. Nueva versión: v20251111_153045",
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

---

### 2. **GET /model/versions** - Listar versiones de modelos

Obtiene todas las versiones de modelos disponibles con sus métricas.

**Requiere:** Autenticación

**Response:**
```json
{
  "models": [
    {
      "version": "v20251111_153045",
      "created_at": "2025-11-11T15:30:45",
      "metrics": {
        "accuracy": 0.95,
        "f1_score": 0.94,
        "precision": 0.93,
        "recall": 0.96,
        "loss": 0.15
      },
      "is_active": true,
      "training_samples": 1500
    },
    {
      "version": "v20251110_120000",
      "created_at": "2025-11-10T12:00:00",
      "metrics": {
        "accuracy": 0.92,
        "f1_score": 0.91,
        "precision": 0.90,
        "recall": 0.93,
        "loss": 0.18
      },
      "is_active": false,
      "training_samples": 1200
    }
  ],
  "active_model": "v20251111_153045"
}
```

---

### 3. **POST /model/activate** - Activar una versión del modelo

Cambia el modelo activo en producción a una versión específica.

**Requiere:** Autenticación + Rol ADMIN

**Request Body:**
```json
{
  "version": "v20251111_153045"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Modelo v20251111_153045 activado exitosamente",
  "active_version": "v20251111_153045"
}
```

---

### 4. **POST /model/upload-dataset** - Subir un dataset

Sube un nuevo archivo CSV para entrenamiento.

**Requiere:** Autenticación + Rol ADMIN

**Request:** Multipart/form-data con archivo CSV

**Formato del CSV:**
El archivo debe contener las siguientes columnas:
- `HbA1c`: Nivel de hemoglobina glicosilada (float)
- `AGE`: Edad del paciente (int)
- `BMI`: Índice de masa corporal (float)
- `Gender`: Género del paciente ('M' o 'F')
- `CLASS`: Clasificación ('N', 'P', 'Y' o 'Negative', 'Prediabetes', 'Diabetes')

**Ejemplo CSV:**
```csv
HbA1c,AGE,BMI,Gender,CLASS
5.2,45,24.5,M,N
6.5,55,28.3,F,P
7.8,62,32.1,M,Y
```

**Response:**
```json
{
  "success": true,
  "message": "Dataset 'nuevo_dataset.csv' subido exitosamente",
  "filename": "nuevo_dataset.csv",
  "rows": 1500,
  "columns": ["HbA1c", "AGE", "BMI", "Gender", "CLASS"]
}
```

---

### 5. **GET /model/active** - Obtener modelo activo

Obtiene información del modelo actualmente en uso.

**Requiere:** Autenticación

**Response:**
```json
{
  "version": "v20251111_153045",
  "created_at": "2025-11-11T15:30:45",
  "metrics": {
    "accuracy": 0.95,
    "f1_score": 0.94,
    "precision": 0.93,
    "recall": 0.96,
    "loss": 0.15
  },
  "training_samples": 1500
}
```

---

### 6. **DELETE /model/version/{version}** - Eliminar una versión

Elimina una versión específica del modelo (no puede ser el modelo activo).

**Requiere:** Autenticación + Rol ADMIN

**Response:**
```json
{
  "success": true,
  "message": "Versión v20251110_120000 eliminada exitosamente"
}
```

---

## 📊 Métricas del Modelo

Cada modelo incluye las siguientes métricas de evaluación:

- **Accuracy (Precisión)**: Porcentaje de predicciones correctas
- **F1-Score**: Media armónica entre precisión y recall
- **Precision**: Proporción de predicciones positivas correctas
- **Recall**: Proporción de casos positivos identificados correctamente
- **Loss**: Pérdida del modelo durante el entrenamiento

---

## 🔄 Flujo de Trabajo Recomendado

### Para Frontend:

1. **Listar modelos disponibles:**
   ```javascript
   GET /model/versions
   ```

2. **Mostrar modelo activo y sus métricas:**
   ```javascript
   GET /model/active
   ```

3. **Reentrenar con datos de la BD:**
   ```javascript
   POST /model/retrain
   {
     "use_database": true,
     "epochs": 190
   }
   ```

4. **Subir nuevo dataset:**
   ```javascript
   POST /model/upload-dataset
   // Enviar archivo CSV
   ```

5. **Reentrenar con nuevo dataset:**
   ```javascript
   POST /model/retrain
   {
     "use_database": false,
     "dataset_name": "nuevo_dataset.csv",
     "epochs": 200
   }
   ```

6. **Activar modelo con mejores métricas:**
   ```javascript
   POST /model/activate
   {
     "version": "v20251111_153045"
   }
   ```

7. **Limpiar versiones antiguas:**
   ```javascript
   DELETE /model/version/v20251110_120000
   ```

---

## 🎨 Ejemplo de UI Recomendada

### Pantalla de Gestión de Modelos:

```
┌─────────────────────────────────────────────────────────┐
│  📊 Gestión de Modelos de ML                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  🟢 Modelo Activo: v20251111_153045                     │
│  📈 Accuracy: 95% | F1-Score: 0.94                      │
│  📅 Creado: 11/11/2025 15:30                            │
│                                                          │
│  [🔄 Reentrenar] [📤 Subir Dataset]                     │
│                                                          │
├─────────────────────────────────────────────────────────┤
│  📋 Versiones Disponibles:                              │
│                                                          │
│  ✅ v20251111_153045 (Activo)                           │
│     Accuracy: 95% | F1: 0.94 | Muestras: 1500          │
│     [Ver detalles]                                       │
│                                                          │
│  ⭕ v20251110_120000                                     │
│     Accuracy: 92% | F1: 0.91 | Muestras: 1200          │
│     [Activar] [Eliminar]                                 │
│                                                          │
│  ⭕ v20251109_093000                                     │
│     Accuracy: 90% | F1: 0.89 | Muestras: 1000          │
│     [Activar] [Eliminar]                                 │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Modal de Reentrenamiento:

```
┌─────────────────────────────────────────────┐
│  🔄 Reentrenar Modelo                       │
├─────────────────────────────────────────────┤
│                                              │
│  Fuente de Datos:                            │
│  ○ Base de Datos (1500 registros)           │
│  ○ Dataset CSV                               │
│    └─ [Seleccionar archivo...]              │
│                                              │
│  Parámetros Avanzados:                       │
│  Épocas: [190]                               │
│  Batch Size: [64]                            │
│  Learning Rate: [0.001]                      │
│  Hidden Size: [64]                           │
│                                              │
│  [Cancelar] [🚀 Iniciar Entrenamiento]      │
│                                              │
└─────────────────────────────────────────────┘
```

---

## ⚠️ Consideraciones Importantes

1. **Solo administradores** pueden reentrenar, activar o eliminar modelos
2. **No se puede eliminar** el modelo activo
3. **El reentrenamiento** puede tardar varios minutos dependiendo del tamaño de los datos
4. **Los datos de la BD** deben tener al menos 10 registros con clasificación
5. **Los archivos CSV** deben seguir el formato especificado
6. **Las versiones** se nombran automáticamente con timestamp

---

## 🔐 Headers Requeridos

Todos los endpoints requieren el header de autenticación:

```
Authorization: Bearer {token}
```

El token se obtiene del endpoint `/token` con credenciales válidas.

---

## 💡 Tips para el Frontend

### Mostrar Progreso de Entrenamiento:
El entrenamiento puede tardar, muestra un indicador de carga y usa WebSockets o polling para actualizar el estado.

### Comparar Modelos:
Crea una tabla comparativa de métricas entre versiones para ayudar al usuario a elegir el mejor modelo.

### Validar CSV antes de subir:
Valida las columnas del CSV en el cliente antes de enviar para evitar errores.

### Confirmación antes de activar:
Muestra las métricas del modelo actual vs el que se va a activar antes de confirmar.

### Gráficos de Métricas:
Usa Chart.js o similar para visualizar la evolución de las métricas entre versiones.

---

¡Listo para integrar! 🚀
