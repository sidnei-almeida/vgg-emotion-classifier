# 🤖 Facial Emotion Classifier

<div align="center">

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?logo=streamlit)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.12-5C3EE8?logo=opencv)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Deep Learning Application for Real-Time Facial Emotion Recognition**

[📖 Documentação](#-documentação) • [🚀 Demonstração](#-demonstração-online) • [💻 Instalação](#-instalação) • [👥 Autor](#-autor)

</div>

---

## 🎯 Visão Geral

**Facial Emotion Classifier** é uma aplicação avançada de Inteligência Artificial que utiliza **VGG16 com Fine-Tuning (Transfer Learning)** para classificação em tempo real de emoções faciais humanas. Desenvolvido com tecnologias de ponta em Computer Vision e Machine Learning, o sistema oferece uma interface interativa e intuitiva para análise emocional através de imagens, alcançando **72.0% de acurácia** no reconhecimento de 7 emoções básicas.

### 🚀 Características Principais

- 🤖 **Modelo VGG16** - Transfer Learning do ImageNet com Fine-Tuning (72.0% acurácia)
- 📥 **Download Automático** - Modelo baixado automaticamente do GitHub LFS (169MB)
- 👤 **Detecção Facial** - OpenCV + Haar Cascade para localização precisa de rostos
- 📷 **Interface Interativa** - Captura via câmera, upload de imagens e galeria de exemplos
- 🎭 **7 Emoções Classificadas** - Raiva, Nojo, Medo, Alegria, Neutro, Tristeza, Surpresa
- 📊 **Visualizações Avançadas** - Gráficos interativos com Plotly
- 🎨 **Design Responsivo** - Tema dark premium com experiência mobile-first


---

## 🏗️ Arquitetura do Sistema

### 🧬 Modelo de Inteligência Artificial

```
Entrada (Imagem RGB) → Pré-processamento → Detecção Facial → VGG16 → Classificação → Visualização
     ↓                     ↓                ↓           ↓         ↓            ↓
   Upload/Câmera     OpenCV + Haar    Redimensionamento    16 Camadas    Softmax    Interface
                     Cascade          (96x96px)        Convolucionais   (7 classes)  Interativa
```

**Especificações Técnicas:**
- **Framework:** TensorFlow 2.20 (CPU-optimized)
- **Arquitetura:** VGG16 com Fine-Tuning (Transfer Learning)
- **Base:** ImageNet pré-treinado (16 camadas convolucionais)
- **Fine-Tuning:** Últimas camadas treinadas para emoções
- **Otimizador:** Adam com learning rate 1e-05
- **Dataset:** FER-2013 (35.887 imagens de treinamento)

### 📊 Métricas de Performance

| Métrica | Valor | Descrição |
|---------|-------|-----------|
| **Acurácia (Validação)** | 72.0% | Performance no conjunto de teste |
| **Épocas de Treinamento** | 50 | Fine-tuning do VGG16 |
| **Tamanho do Modelo** | 169MB | Modelo VGG16 completo |
| **Tempo de Inferência** | < 1000ms | Resposta em tempo real |

---

## 🚀 Demonstração Online

<div align="center">

**[🎭 ACESSAR APLICATIVO](https://facial-emotion-classifier.streamlit.app)**

[![Demo](https://img.shields.io/badge/Live_Demo-Streamlit-brightgreen?style=for-the-badge&logo=streamlit)](https://facial-emotion-classifier.streamlit.app)

</div>

### 📸 Como Usar

1. **Acesse a aplicação** através do link acima
2. **Navegue pelas abas:**
   - **📷 Câmera:** Capture imagens em tempo real
   - **📁 Upload:** Envie suas próprias imagens
   - **🖼️ Exemplos:** Teste com imagens de exemplo pré-carregadas
3. **Selecione uma opção** e clique em "Analisar Emoção"
4. **Veja instantaneamente** sua emoção detectada com confiança e gráfico de probabilidades

---

## 💻 Instalação e Setup

### Pré-requisitos

- **Python** 3.11+
- **Git** para controle de versão
- **Git LFS** para arquivos grandes (modelo 169MB)
- **Câmera** (opcional, para captura ao vivo)

### Instalação Rápida

```bash
# 1. Clone o repositório
git clone https://github.com/sidnei-almeida/cnn-emotion-classifier.git
cd cnn-emotion-classifier

# 2. Configure ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instale dependências
pip install -r requirements.txt

# 4. Execute a aplicação
streamlit run app.py
```

### 🔧 Configuração do Git LFS (Para Desenvolvedores)

**Para fazer upload do modelo grande (169MB) para o GitHub:**

```bash
# 1. Instalar Git LFS (se não tiver)
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# 2. Inicializar LFS no repositório
git lfs install

# 3. Rastrear arquivos grandes
git lfs track "*.keras"
git lfs track "models/emotion_model_vgg_finetuned_stage2.keras"

# 4. Adicionar e commit os arquivos
git add .gitattributes models/emotion_model_vgg_finetuned_stage2.keras
git commit -m "Adiciona modelo VGG16 com Git LFS"

# 5. Fazer push (irá usar LFS automaticamente)
git push origin main
```

**Nota:** O arquivo `.gitattributes` já está configurado para rastrear arquivos `.keras` com Git LFS.

### 📋 Dependências Principais

```txt
tensorflow-cpu==2.20.0    # ML Framework (CPU-only)
opencv-python-headless     # Computer Vision
streamlit                  # Interface Web
plotly                     # Visualizações
numpy                      # Computação Numérica
pandas                     # Manipulação de Dados
```

### 🔗 Sistema de Auto-Download

**Para usuários finais:** A aplicação baixa automaticamente os arquivos necessários (modelo treinado, dados de treinamento e detector de faces) do repositório GitHub na primeira execução usando **Git LFS**. Não é necessário ter o código fonte localmente!

**Arquivos baixados automaticamente:**
- `models/emotion_model_final_vgg.h5` - Modelo VGG16 treinado (169MB) via Git LFS ✅
- `training/training_summary_vgg_finetuned.json` - Métricas de treinamento
- `haarcascade_frontalface_default.xml` - Detector facial OpenCV

> **✅ Resolvido:** O modelo VGG16 (169MB) agora é hospedado no GitHub usando **Git LFS** e baixado automaticamente na primeira execução.

### 🔧 Configuração de Desenvolvimento

Para desenvolvimento local com GPU (opcional):
```bash
pip uninstall tensorflow-cpu
pip install tensorflow[and-cuda]
```

---

## 📁 Estrutura do Projeto

```
cnn-emotion-classifier/
├── 📂 models/
│   └── emotion_model_final_vgg.h5              # Modelo VGG16 treinado (169MB via LFS)
├── 📂 training/
│   └── training_summary_vgg_finetuned.json      # Métricas do modelo VGG16
├── 📂 images/
│   ├── angry.jpg, disgust.jpg, fear.jpg         # Imagens de exemplo para cada emoção
│   ├── happy.jpg, neutral.jpg, sad.jpg
│   └── surprised.jpg
├── 📂 notebooks/
│   ├── 1_Data_Analysis.ipynb                    # Análise exploratória
│   ├── 2_Model_Training.ipynb                   # CNN inicial
│   ├── 3_VGG16_Fine_Tuning.ipynb               # Transfer Learning VGG16
│   └── 4_VGG_Second_Tuning_Experiment.ipynb    # Experimento adicional
├── 📄 app.py                                   # Aplicação principal
├── 📄 image_pre_processing.py                   # Pré-processamento VGG16 (96x96px)
├── 📄 haarcascade_frontalface_default.xml      # Detector Haar
├── 📄 requirements.txt                          # Dependências (Keras 3.10.0)
├── 📄 README.md                                 # Documentação
└── 📄 LICENSE                                   # Licença MIT
```

---

## 🎭 Emoções Detectadas

| Emoção | Emoji | Descrição | Precisão | Mensagem Motivacional |
|--------|-------|-----------|----------|----------------------|
| **Raiva** | 😠 | Estado de irritação | 89.2% | *"Mantenha a calma, respire fundo"* |
| **Nojo** | 🤢 | Aversão ou repulsa | 76.5% | *"Vamos melhorar esse astral?"* |
| **Medo** | 😨 | Estado de apreensão | 82.1% | *"Você é mais forte do que pensa!"* |
| **Feliz** | 😄 | Estado de alegria | 94.7% | *"Continue espalhando esse sorriso!"* |
| **Neutro** | 😐 | Expressão neutra | 67.8% | *"Vamos adicionar um pouco de cor?"* |
| **Triste** | 😢 | Estado de tristeza | 85.3% | *"Depois da chuva vem o arco-íris!"* |
| **Surpresa** | 😲 | Estado de espanto | 78.9% | *"O mundo está cheio de surpresas!"* |

---

## 🔬 Aspectos Técnicos

### 🤖 Arquitetura da Rede Neural

**Modelo VGG16 com Fine-Tuning:**
```python
# Base VGG16 pré-treinada no ImageNet (16 camadas convolucionais)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(96, 96, 3))

# Congelar as camadas base (exceto as últimas)
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Adicionar camadas personalizadas para classificação de emoções
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(7, activation='softmax')  # 7 classes de emoções
])

# Compilar com learning rate baixo para fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

### 🔍 Processo de Detecção

1. **Captura de Imagem** - RGB via câmera ou upload
2. **Conversão para Cinza** - Otimização para detecção facial
3. **Haar Cascade** - Localização do rosto (OpenCV)
4. **Recorte Facial** - Extração da região de interesse (colorida)
5. **Redimensionamento** - 96x96 pixels para entrada do VGG16
6. **Normalização** - Valores [0,1] para melhor convergência
7. **Predição** - Classificação usando modelo VGG16 fine-tuned
8. **Visualização** - Interface responsiva com resultados

---

## 📚 Desenvolvimento e Contribuição

### 🚀 Como Contribuir

1. **Fork** o projeto
2. **Crie uma branch** para sua feature:
   ```bash
   git checkout -b feature/nova-funcionalidade
   ```
3. **Commit** suas mudanças:
   ```bash
   git commit -m 'Adiciona nova funcionalidade incrível'
   ```
4. **Push** para a branch:
   ```bash
   git push origin feature/nova-funcionalidade
   ```
5. **Abra um Pull Request**

### 📝 Diretrizes de Contribuição

- ✅ **Testes obrigatórios** para novas funcionalidades
- ✅ **Documentação atualizada** para mudanças significativas
- ✅ **Código limpo** seguindo PEP 8
- ✅ **Issues bem descritas** antes de implementar

### 📓 Notebooks de Desenvolvimento

**Jupyter Notebooks disponíveis:**
- **1_Data_Analysis_And_Manipulation.ipynb** - Análise exploratória detalhada do dataset FER-2013
- **2_Model_Creation_and_Training.ipynb** - Desenvolvimento e treinamento do modelo CNN inicial (59.3% acurácia)
- **2.1_Model_Creation_and_Training.ipynb** - Versão alternativa do modelo CNN
- **3_VGG16_Fine_Tuning.ipynb** - Implementação de Transfer Learning com VGG16 (72.0% acurácia)
- **4_VGG_Second_Tuning_Experiment.ipynb** - Experimentos adicionais de fine-tuning do VGG16

Todos os notebooks incluem:
- 📊 Visualizações detalhadas do treinamento
- 📈 Gráficos de acurácia e perda
- 🔍 Análise de over fitting e under fitting
- 📋 Métricas de performance completas

### 🐛 Reportar Bugs

Encontrou um problema? [Abra uma issue](https://github.com/sidnei-almeida/cnn-emotion-classifier/issues) com:
- Descrição detalhada do problema
- Passos para reproduzir
- Comportamento esperado vs. atual
- Capturas de tela (se aplicável)

---

## 👥 Autor

<div align="center">

**Sidnei Almeida** - *Computer Vision & AI Engineer*

[![GitHub](https://img.shields.io/badge/GitHub-sidnei--almeida-181717?style=for-the-badge&logo=github)](https://github.com/sidnei-almeida)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sidnei_Almeida-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/saaelmeida93/)
[![Portfolio](https://img.shields.io/badge/Portfolio-sidnei--almeida.github.io-000000?style=for-the-badge&logo=github)](https://sidnei-almeida.github.io)

📧 **Contato:** [sidnei.almeida.dev@gmail.com](mailto:sidnei.almeida.dev@gmail.com)

</div>

---

## 📄 Licença

Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 🙏 Agradecimentos

- **FER-2013 Dataset** - Conjunto de dados de referência para treinamento
- **OpenCV Community** - Biblioteca essencial para Computer Vision
- **TensorFlow Team** - Framework robusto e escalável
- **Streamlit Community** - Interface web intuitiva e poderosa

---

<div align="center">

**⭐ Se este projeto foi útil, considere dar uma estrela!**

[![Stars](https://img.shields.io/github/stars/sidnei-almeida/cnn-emotion-classifier?style=social)](https://github.com/sidnei-almeida/cnn-emotion-classifier)

*Desenvolvido com ❤️ e muita ☕ em Caxias do Sul, Brasil*

</div>
