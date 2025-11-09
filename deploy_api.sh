#!/bin/bash

# Script para deploy da API no Hugging Face Spaces

set -e

echo "🚀 Preparando deploy da API para Hugging Face Spaces..."

# Verificar se está em um repositório git
if [ ! -d ".git" ]; then
    echo "❌ Erro: Este diretório não é um repositório git"
    echo "   Execute: git init"
    exit 1
fi

# Verificar se os arquivos necessários existem
echo "📋 Verificando arquivos necessários..."

REQUIRED_FILES=(
    "api.py"
    "Dockerfile"
    "requirements-api.txt"
    "app.yaml"
    "image_pre_processing.py"
    "haarcascade_frontalface_default.xml"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Arquivo não encontrado: $file"
        exit 1
    fi
done

# Verificar se há modelos
if [ ! -d "models" ] || [ -z "$(ls -A models/*.h5 models/*.keras 2>/dev/null)" ]; then
    echo "⚠️  Aviso: Nenhum modelo encontrado na pasta models/"
    echo "   Certifique-se de fazer upload dos modelos antes do deploy"
fi

echo "✅ Todos os arquivos necessários estão presentes"
echo ""
echo "📝 Próximos passos:"
echo "1. Crie um novo Space no Hugging Face: https://huggingface.co/spaces"
echo "2. Escolha 'Docker' como SDK"
echo "3. Clone o repositório do Space:"
echo "   git clone https://huggingface.co/spaces/seu-usuario/seu-space"
echo "4. Copie os arquivos para o diretório do Space"
echo "5. Faça commit e push:"
echo "   git add ."
echo "   git commit -m 'Deploy API'"
echo "   git push"
echo ""
echo "📚 Consulte README_API.md para mais informações"

