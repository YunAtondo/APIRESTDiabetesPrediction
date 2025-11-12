import torch
from datetime import datetime
from sqlalchemy.orm import Session
from ..models.registrosModel import Registro
from ..models.recomendacionesPreviasModel import Recomendacion
from ..utils.model_loader import load_model_and_scaler
import os

# Obtener ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(BASE_DIR, "mlModels", "best_diabetes_model.pth")
scaler_path = os.path.join(BASE_DIR, "mlModels", "scaler.pkl")

# Variables globales para lazy loading (singleton pattern)
_model = None
_scaler = None

def get_model_and_scaler():
    """Lazy loading del modelo - solo carga una vez cuando se necesita"""
    global _model, _scaler
    if _model is None or _scaler is None:
        print("🔄 Cargando modelo de predicción...")
        _model, _scaler = load_model_and_scaler(model_path, scaler_path)
        print("✅ Modelo cargado exitosamente!")
    return _model, _scaler

# Diccionario de clases
CLASES = {0: "Negative", 1: "Prediabetes", 2: "Diabetes"}

def generar_recomendaciones(data) -> list[str]:
    recomendaciones = []

    # Recomendaciones para BMI
    if data.BMI < 18.5:
        recomendaciones.append("Tu IMC indica bajo peso. Podrías necesitar una dieta más balanceada y calórica. Consulta a un nutricionista.")
    elif 18.5 <= data.BMI <= 24.9:
        recomendaciones.append("Tu IMC está dentro del rango saludable. ¡Sigue así!")
    elif 25 <= data.BMI <= 29.9:
        recomendaciones.append("Tu IMC indica sobrepeso. Considera hacer ejercicio regular y revisar tu dieta.")
    elif 30 <= data.BMI <= 34.9:
        recomendaciones.append("Tu IMC indica obesidad grado I. Es recomendable comenzar un plan de alimentación y actividad física.")
    elif 35 <= data.BMI <= 39.9:
        recomendaciones.append("Tu IMC indica obesidad grado II. Deberías buscar asesoría médica y nutricional.")
    elif data.BMI >= 40:
        recomendaciones.append("Tu IMC indica obesidad grado III. Es importante actuar con apoyo médico y seguimiento especializado.")

    # Recomendaciones para HbA1c
    if data.HbA1c < 5.7:
        recomendaciones.append("Tu nivel de HbA1c está en un rango saludable.")
    elif 5.7 <= data.HbA1c <= 6.4:
        recomendaciones.append("Tu nivel de HbA1c sugiere prediabetes. Mantén hábitos saludables y monitorea tu estado.")
    elif data.HbA1c > 6.4:
        recomendaciones.append("Tu nivel de HbA1c es alto. Podrías estar desarrollando diabetes. Es recomendable consultar a un médico.")

    # Recomendaciones por edad
    if data.AGE < 18:
        recomendaciones.append("Como menor de edad, es importante que un adulto supervise tus hábitos de salud.")
    elif 18 <= data.AGE <= 40:
        recomendaciones.append("A esta edad, mantener una vida activa y una dieta equilibrada es clave para la prevención.")
    elif 41 <= data.AGE <= 60:
        recomendaciones.append("Es importante realizar chequeos médicos regulares y mantener hábitos saludables.")
    elif data.AGE > 60:
        recomendaciones.append("En esta etapa de la vida, los controles de salud son fundamentales para el bienestar general.")

    return recomendaciones

def recomendacion_por_clasificacion(clase: str) -> str:
    if clase == "Negative":
        return "Actualmente no se detectan indicios de diabetes. Continúa con tus hábitos saludables y realiza chequeos periódicos."
    elif clase == "Prediabetes":
        return "Se detectan signos de prediabetes. Es un buen momento para mejorar la alimentación, hacer ejercicio y evitar el sedentarismo."
    elif clase == "Diabetes":
        return "Se ha detectado diabetes. Es fundamental que consultes a un profesional de la salud para iniciar un tratamiento adecuado."
    return "Clasificación desconocida. Por favor realiza una nueva evaluación o consulta con un profesional."



def clasificar_y_guardar(data, db: Session):
    # Cargar modelo solo cuando se necesita (lazy loading)
    model, scaler = get_model_and_scaler()
    
    gender_encoded = 1 if data.Gender.lower() == "M" else 0
    entrada = [[data.HbA1c, data.BMI, data.AGE, gender_encoded]]

    entrada_escalada = scaler.transform(entrada)
    entrada_tensor = torch.tensor(entrada_escalada, dtype=torch.float32)

    with torch.no_grad():
        salida = model(entrada_tensor)
        pred = torch.argmax(salida, dim=1).item()

    clase = CLASES.get(pred, "Desconocido")

    # Guardar en la base de datos
    nuevo_registro = Registro(
        id_usuario=data.id_usuario,
        fecha_registro=datetime.now(),
        AGE=data.AGE,
        Gender=data.Gender,
        BMI=str(data.BMI),
        HbA1c=str(data.HbA1c),
        CLASS=clase
    )
    db.add(nuevo_registro)
    db.commit()
    db.refresh(nuevo_registro)
    
    # Generar y guardar recomendaciones
    recomendaciones = generar_recomendaciones(data)
    recomendacion_final = recomendacion_por_clasificacion(clase)
    recomendaciones.append(recomendacion_final)
    
    if data.id_usuario is not None:
        mensaje_completo = " ".join(recomendaciones)

        nueva_recomendacion = Recomendacion(
            id_usuario=data.id_usuario,
            id_registro=nuevo_registro.id,
            recomendacion=mensaje_completo[:500],  # asegurarse que no exceda el límite
            fecha_generada=datetime.now(),
            is_active="ACTIVO"
        )
        db.add(nueva_recomendacion)

        db.commit()

    return clase, nuevo_registro.id, recomendaciones