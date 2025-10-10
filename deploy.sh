#!/bin/bash

# Script de deploy para GitHub com Git LFS
# Uso: ./deploy.sh

echo "🚀 Iniciando deploy do Facial Emotion Classifier..."

# 1. Verificar se Git LFS está instalado
if ! command -v git-lfs &> /dev/null; then
    echo "❌ Git LFS não encontrado. Instalando..."
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install git-lfs -y
fi

# 2. Inicializar Git LFS
echo "🔧 Inicializando Git LFS..."
git lfs install

# 3. Verificar arquivos grandes
echo "📊 Verificando arquivos grandes..."
git lfs ls-files

# 4. Adicionar arquivos ao Git
echo "📁 Adicionando arquivos ao Git..."
git add .

# 5. Fazer commit
echo "💾 Fazendo commit..."
git commit -m "feat: Atualiza aplicação com modelo VGG16 (72.4% acurácia)

- Modelo VGG16 com Transfer Learning
- Acurácia de 72.4% no conjunto de teste
- Pré-processamento otimizado para 96x96px
- Sistema de auto-download funcional
- Configuração Git LFS para arquivos grandes"

# 6. Fazer push
echo "⬆️  Fazendo push para GitHub..."
git push origin main

echo "✅ Deploy concluído!"
echo "🎉 Aplicação disponível em: https://facial-emotion-classifier.streamlit.app"
echo "📁 Repositório: https://github.com/sidnei-almeida/cnn-emotion-classifier"
