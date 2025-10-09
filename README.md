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

**Facial Emotion Classifier** é uma aplicação avançada de Inteligência Artificial que utiliza **Convolutional Neural Networks (CNN)** para classificação em tempo real de emoções faciais humanas. Desenvolvido com tecnologias de ponta em Computer Vision e Machine Learning, o sistema oferece uma interface interativa e intuitiva para análise emocional através de imagens.

### ✨ Características Principais

- 🧠 **Modelo CNN Otimizado** - Arquitetura personalizada com 59.3% de acurácia
- 👤 **Detecção Facial Automática** - OpenCV + Haar Cascade para localização precisa
- 📷 **Interface Interativa** - Captura via câmera e upload de imagens
- 🎭 **7 Emoções Classificadas** - Raiva, Nojo, Medo, Alegria, Neutro, Tristeza, Surpresa
- 📊 **Visualizações Avançadas** - Gráficos interativos com Plotly
- 🎨 **Design Responsivo** - Tema dark premium com experiência mobile-first

---

## 🏗️ Arquitetura do Sistema

### 🧬 Modelo de Inteligência Artificial

```
Entrada (Imagem RGB) → Pré-processamento → Detecção Facial → CNN → Classificação → Visualização
     ↓                     ↓                ↓           ↓         ↓            ↓
   Upload/Câmera     OpenCV + Haar    Redimensionamento    3 Camadas     Softmax    Interface
                     Cascade          (48x48px)        Convolucionais   (7 classes)  Interativa
```

**Especificações Técnicas:**
- **Framework:** TensorFlow 2.20 (CPU-optimized)
- **Arquitetura:** CNN Sequencial com 3 blocos convolucionais
- **Camadas:** Conv2D → BatchNorm → MaxPool → Dropout
- **Otimizador:** Adam com learning rate adaptativo
- **Dataset:** FER-2013 (35.887 imagens de treinamento)

### 📊 Métricas de Performance

| Métrica | Valor | Descrição |
|---------|-------|-----------|
| **Acurácia (Validação)** | 59.3% | Performance no conjunto de teste |
| **Épocas de Treinamento** | 51 | Early stopping automático |
| **Tamanho do Modelo** | 1.2MB | Otimizado para deploy |
| **Tempo de Inferência** | < 500ms | Resposta em tempo real |

---

## 🚀 Demonstração Online

<div align="center">

**[🎭 ACESSAR APLICATIVO](https://facial-emotion-classifier.streamlit.app)**

[![Demo](https://img.shields.io/badge/Live_Demo-Streamlit-brightgreen?style=for-the-badge&logo=streamlit)](https://facial-emotion-classifier.streamlit.app)

</div>

### 📸 Como Usar

1. **Acesse a aplicação** através do link acima
2. **Clique em "📸 Iniciar Câmera"** na aba "Detector"
3. **Permita acesso à câmera** no seu navegador
4. **Aponte para seu rosto** e clique no botão de captura
5. **Veja instantaneamente** sua emoção detectada com confiança

---

## 💻 Instalação e Setup

### Pré-requisitos

- **Python** 3.11+
- **Git** para controle de versão
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

### 📋 Dependências Principais

```txt
tensorflow-cpu==2.20.0    # ML Framework (CPU-only)
opencv-python-headless     # Computer Vision
streamlit                  # Interface Web
plotly                     # Visualizações
numpy                      # Computação Numérica
pandas                     # Manipulação de Dados
```

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
│   └── emotion_model.keras          # Modelo CNN treinado
├── 📂 training/
│   └── training_summary.json        # Métricas de treinamento
├── 📂 notebooks/
│   ├── 1_Data_Analysis.ipynb        # Análise exploratória
│   └── 2_Model_Training.ipynb       # Processo de treinamento
├── 📂 src/
│   ├── app.py                       # Aplicação principal
│   ├── image_preprocessing.py       # Pré-processamento facial
│   └── haarcascade_frontalface_default.xml  # Detector Haar
├── 📄 requirements.txt              # Dependências
├── 📄 README.md                     # Documentação
└── 📄 LICENSE                       # Licença MIT
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

```python
model = Sequential([
    # Bloco 1: Extração de características básicas
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.2),

    # Bloco 2: Características intermediárias
    Conv2D(64, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.3),

    # Bloco 3: Características avançadas
    Conv2D(128, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.4),

    # Classificação final
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(7, activation='softmax')
])
```

### 🔍 Processo de Detecção

1. **Captura de Imagem** - RGB via câmera ou upload
2. **Conversão para Cinza** - Otimização para detecção facial
3. **Haar Cascade** - Localização do rosto (OpenCV)
4. **Recorte Facial** - Extração da região de interesse
5. **Redimensionamento** - 48x48 pixels para entrada da CNN
6. **Normalização** - Valores [0,1] para melhor convergência
7. **Predição** - Classificação em tempo real
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

*Desenvolvido com ❤️ e muita ☕ em São Paulo, Brasil*

</div>
