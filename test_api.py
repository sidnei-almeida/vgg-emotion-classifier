"""
Script de teste para a API de classificação de emoções
"""

import requests
import json
from pathlib import Path

API_URL = "http://localhost:8000"

def test_health():
    """Testa o endpoint de health"""
    print("🔍 Testando /health...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Resposta: {json.dumps(response.json(), indent=2)}")
    print()

def test_predict(image_path: str):
    """Testa o endpoint de predição"""
    print(f"🔍 Testando /predict com {image_path}...")
    
    if not Path(image_path).exists():
        print(f"❌ Arquivo não encontrado: {image_path}")
        return
    
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        response = requests.post(f"{API_URL}/predict", files=files)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Emoção detectada: {result['emotion']}")
        print(f"Confiança: {result['confidence']:.2%}")
        print(f"Mensagem: {result['message']}")
        print("\nProbabilidades:")
        for emotion, prob in sorted(result['probabilities'].items(), 
                                   key=lambda x: x[1], reverse=True):
            print(f"  {emotion}: {prob:.2%}")
    else:
        print(f"Erro: {response.text}")
    print()

def test_emotions():
    """Testa o endpoint de emoções"""
    print("🔍 Testando /emotions...")
    response = requests.get(f"{API_URL}/emotions")
    print(f"Status: {response.status_code}")
    print(f"Emoções disponíveis: {response.json()['emotions']}")
    print()

if __name__ == "__main__":
    print("=" * 50)
    print("Testes da API de Classificação de Emoções")
    print("=" * 50)
    print()
    
    # Test health
    try:
        test_health()
    except Exception as e:
        print(f"❌ Erro ao testar health: {e}")
        print("Certifique-se de que a API está rodando em http://localhost:8000")
        exit(1)
    
    # Test emotions
    test_emotions()
    
    # Test prediction (se houver imagem de exemplo)
    example_images = [
        "images/happy.jpg",
        "images/sad.jpg",
        "images/angry.jpg"
    ]
    
    for img_path in example_images:
        if Path(img_path).exists():
            test_predict(img_path)
            break
    else:
        print("⚠️  Nenhuma imagem de exemplo encontrada para teste")
        print("   Use: python test_api.py <caminho_para_imagem>")
        print()

