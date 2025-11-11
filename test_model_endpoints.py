"""
Script de prueba rápida para los endpoints de gestión de modelos
Ejecutar después de iniciar el servidor
"""
import requests
import json

# Configuración
BASE_URL = "http://localhost:8000"
# Primero necesitas obtener un token de autenticación como ADMIN
# Usa las credenciales de admin que creaste

print("=" * 60)
print("🧪 PRUEBA DE ENDPOINTS DE GESTIÓN DE MODELOS")
print("=" * 60)

# 1. Login como admin (necesitas tener un usuario admin creado)
print("\n1️⃣ Obteniendo token de autenticación...")
login_response = requests.post(
    f"{BASE_URL}/token",
    data={
        "username": "admin@admin.com",  # Ajusta según tu admin
        "password": "admin1234"          # Ajusta según tu admin
    }
)

if login_response.status_code == 200:
    token = login_response.json()["access_token"]
    print("✅ Token obtenido exitosamente")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # 2. Listar versiones de modelos
    print("\n2️⃣ Listando versiones de modelos...")
    versions_response = requests.get(
        f"{BASE_URL}/model/versions",
        headers=headers
    )
    
    if versions_response.status_code == 200:
        data = versions_response.json()
        print(f"✅ Modelos encontrados: {len(data['models'])}")
        print(f"   Modelo activo: {data['active_model']}")
        
        for model in data['models']:
            print(f"\n   📊 {model['version']}")
            print(f"      Accuracy: {model['metrics']['accuracy']:.2%}")
            print(f"      F1-Score: {model['metrics']['f1_score']:.4f}")
            print(f"      Activo: {'✅' if model['is_active'] else '⭕'}")
    else:
        print(f"❌ Error: {versions_response.text}")
    
    # 3. Ver modelo activo
    print("\n3️⃣ Obteniendo información del modelo activo...")
    active_response = requests.get(
        f"{BASE_URL}/model/active",
        headers=headers
    )
    
    if active_response.status_code == 200:
        data = active_response.json()
        print(f"✅ Modelo activo: {data['version']}")
        if 'metrics' in data:
            print(f"   Accuracy: {data['metrics']['accuracy']:.2%}")
            print(f"   F1-Score: {data['metrics']['f1_score']:.4f}")
            print(f"   Precision: {data['metrics']['precision']:.4f}")
            print(f"   Recall: {data['metrics']['recall']:.4f}")
    else:
        print(f"❌ Error: {active_response.text}")
    
    # 4. Probar reentrenamiento (comentado por defecto porque tarda)
    print("\n4️⃣ Reentrenamiento de modelo (comentado)...")
    print("   Para probar el reentrenamiento, descomenta el código en el script")
    
    # DESCOMENTA ESTO PARA PROBAR REENTRENAMIENTO:
    # print("   ⏳ Iniciando reentrenamiento...")
    # retrain_response = requests.post(
    #     f"{BASE_URL}/model/retrain",
    #     headers=headers,
    #     json={
    #         "use_database": True,
    #         "epochs": 50,  # Pocas épocas para prueba rápida
    #         "batch_size": 64
    #     }
    # )
    # 
    # if retrain_response.status_code == 200:
    #     data = retrain_response.json()
    #     print(f"   ✅ Modelo reentrenado: {data['version']}")
    #     print(f"      Accuracy: {data['metrics']['accuracy']:.2%}")
    #     print(f"      F1-Score: {data['metrics']['f1_score']:.4f}")
    #     print(f"      Tiempo: {data['training_time']:.2f}s")
    # else:
    #     print(f"   ❌ Error: {retrain_response.text}")
    
    print("\n" + "=" * 60)
    print("✅ Pruebas completadas")
    print("=" * 60)
    print("\n📚 Para más información, consulta:")
    print("   - README_MODEL_RETRAINING.md")
    print("   - API_MODEL_MANAGEMENT.md")
    print("   - FRONTEND_EXAMPLES.ts")
    
else:
    print(f"❌ Error en login: {login_response.text}")
    print("\n💡 Asegúrate de:")
    print("   1. Tener el servidor corriendo (uvicorn app.main:app --reload)")
    print("   2. Tener un usuario admin creado")
    print("   3. Ajustar las credenciales en este script")
