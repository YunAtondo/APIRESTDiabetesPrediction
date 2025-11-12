#!/usr/bin/env bash
# build.sh para Render

set -o errexit

# Actualizar pip
pip install --upgrade pip

# Instalar dependencias
pip install -r requirements.txt

# Crear directorios necesarios
mkdir -p mlModels/versions

echo "✅ Build completado exitosamente"
