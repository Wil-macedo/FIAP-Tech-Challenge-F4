<div align="center">

# 📈 Tech Challenge - Fase 4

### Previsão de Preços de Ações com LSTM

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Sistema de Machine Learning para previsão de preços de ações utilizando redes neurais LSTM (Long Short-Term Memory), com deploy automatizado e monitoramento em tempo real.**

[Documentação](#-documentação) • [Instalação](#-instalação) • [API](#-api) • [Deploy](#-deploy) • [Monitoramento](#-monitoramento)

</div>

---

## 🎯 Sobre o Projeto

Este projeto foi desenvolvido como parte do **Tech Challenge - Fase 4** da Pós-Tech FIAP em **Machine Learning Engineering**. O objetivo é criar um sistema completo de previsão de preços de ações, desde a coleta de dados até o deploy em produção.

### 🔑 Características Principais

- **🧠 Modelo LSTM**: Rede neural recorrente para capturar padrões temporais
- **📊 MLflow Integration**: Rastreamento completo de experimentos e métricas
- **🚀 API RESTful**: Endpoints Flask com documentação Swagger
- **📦 Docker**: Containerização para deploy simplificado
- **☁️ Cloud Ready**: Deploy em AWS EC2
- **📈 Monitoramento**: Tracking de performance e uso de recursos em tempo real
- **⚡ Alta Performance**: Compressão de resposta e caching de modelo

---

## 🏗️ Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                     API Flask (Port 8010)                    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   /predict   │  │   /monitor   │  │   /mlflow    │      │
│  │  Predições   │  │ Métricas RT  │  │  UI MLflow   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Modelo LSTM (Keras)                       │
├─────────────────────────────────────────────────────────────┤
│  • 2 Camadas LSTM (50 neurônios cada)                       │
│  • Dropout (0.2) para regularização                         │
│  • Input: 60 dias de histórico                              │
│  • Output: Previsão do próximo dia                          │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               Dados Históricos (Yahoo Finance)               │
│                   Symbol: MSFT (2010-2025)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 Requisitos

- **Python**: 3.8 ou superior
- **Docker**: 20.10+ (opcional, para containerização)
- **Memória**: Mínimo 4GB RAM
- **Espaço em Disco**: ~2GB para modelo e dependências

---

## 🚀 Instalação

### Opção 1: Instalação Local

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/Tech-Challenge-F4.git
cd Tech-Challenge-F4

# Crie um ambiente virtual
python -m venv venv

# Ative o ambiente virtual
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Instale as dependências
pip install -r requirements.txt

# Treine o modelo (primeira vez)
python modelTrain.py

# Inicie a aplicação
python app.py
```

A API estará disponível em: `http://localhost:8010`

### Opção 2: Docker

```bash
# Construa a imagem
docker build -t tech_challenge_f4 .

# Execute o container
docker run -d -p 8010:8010 --name tc_fase_4 tech_challenge_f4
```

### Opção 3: Docker Hub (Imagem Pronta)

```bash
# Baixe e execute a imagem do Docker Hub
docker pull willmacedo1/tc_fase_4
docker run -d --restart=always -p 8010:8010 --name tc_fase_4 willmacedo1/tc_fase_4
```

---

## 📖 Documentação

### Estrutura do Projeto

```
Tech-Challenge-F4/
│
├── app.py                  # API Flask principal
├── modelTrain.py           # Script de treinamento do modelo
├── modelPredict.py         # Módulo de predição
├── processData.py          # Processamento e normalização de dados
├── swagger.yaml            # Documentação OpenAPI
├── requirements.txt        # Dependências Python
├── Dockerfile              # Configuração Docker
│
├── modelFiles/             # Modelos treinados
│   ├── my_model.keras      # Modelo LSTM
│   └── scaler.pkl          # Scaler MinMaxScaler
│
├── mlruns/                 # Experimentos MLflow
├── templates/              # Templates HTML
└── jupyter/                # Notebooks de análise
```

### Pipeline de Dados

1. **Coleta**: Dados históricos via `yfinance` (Yahoo Finance)
2. **Pré-processamento**: Normalização MinMaxScaler (0-1)
3. **Feature Engineering**: Janelas temporais de 60 dias
4. **Treinamento**: LSTM com 80/20 train/test split
5. **Validação**: Métricas MAE, RMSE, R²
6. **Deploy**: Modelo salvo em formato Keras

---

## 🔌 API

### Endpoints Disponíveis

#### 1. Documentação Interativa (Swagger)
```
GET /apidocs
```
Interface Swagger UI para testar os endpoints.

#### 2. Realizar Previsão
```http
POST /predict
Content-Type: application/json

{
  "predict": [
    23.30, 23.31, 23.17, 22.92, 23.08,
    // ... 55 valores adicionais (total: 60)
  ]
}
```

**Resposta:**
```json
{
  "predicted_price": 22.65,
  "response_time_sec": 0.0234,
  "memory_usage_percent": 45.2
}
```

#### 3. Monitoramento do Sistema
```http
GET /monitor
```

**Resposta:**
```json
{
  "CURRENT CPU": 15.3,
  "MEMORY %": 42.8,
  "RESPONSES": [
    {
      "timestamp": "2025-02-06T10:30:00",
      "response_time": 0.0234,
      "memory_usage": 45.2
    }
  ]
}
```

#### 4. MLflow UI
```
GET /mlflow
```
Redireciona para a interface do MLflow (porta 8020).

---

## 🎓 Treinamento do Modelo

### Executar Treinamento

```bash
python modelTrain.py
```

### Hiperparâmetros

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| **Arquitetura** | 2x LSTM + Dense | Camadas recorrentes |
| **Neurônios LSTM** | 50 | Por camada |
| **Dropout** | 0.2 | Regularização |
| **Optimizer** | Adam | Otimizador |
| **Loss** | MSE | Função de perda |
| **Epochs** | 15 | Iterações de treinamento |
| **Batch Size** | 32 | Tamanho do lote |
| **Time Steps** | 60 | Dias de histórico |

### Métricas de Avaliação

O modelo é avaliado com as seguintes métricas:

- **MSE (Mean Squared Error)**: Erro quadrático médio
- **RMSE (Root Mean Squared Error)**: Raiz do erro quadrático médio
- **R² Score**: Coeficiente de determinação

Todas as métricas são automaticamente logadas no **MLflow** para versionamento e comparação.

---

## 🐳 Deploy

### Deploy Local (Development)

```bash
python app.py
```

### Deploy com Docker

```bash
# Build
docker build -t tech_challenge_f4 .

# Tag para Docker Hub
docker tag tech_challenge_f4 seu_usuario/tc_fase_4:latest

# Push para Docker Hub
docker push seu_usuario/tc_fase_4:latest

# Deploy em produção
docker run -d --restart=always -p 8010:8010 --name tc_fase_4 seu_usuario/tc_fase_4:latest
```

### Deploy em AWS EC2

```bash
# 1. Conecte-se à instância EC2
ssh -i sua-chave.pem ubuntu@seu-ip-ec2

# 2. Instale Docker
sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker

# 3. Execute o container
sudo docker pull willmacedo1/tc_fase_4
sudo docker run -d --restart=always -p 8010:8010 --name tc_fase_4 willmacedo1/tc_fase_4

# 4. Verifique o status
sudo docker ps
```

**URL de Produção**: `https://ec2-18-234-186-76.compute-1.amazonaws.com:8010/`

### Auto-start com Systemd (EC2)

Crie o arquivo `/etc/systemd/system/flask_app.service`:

```ini
[Unit]
Description=Flask App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Tech-Challenge-F4
ExecStart=/bin/bash -c 'source /home/ubuntu/Tech-Challenge-F4/venv/bin/activate && python3 /home/ubuntu/Tech-Challenge-F4/app.py'
Restart=always
Environment=PATH=/usr/bin:/usr/local/bin
Environment=FLASK_APP=/home/ubuntu/Tech-Challenge-F4/app.py

[Install]
WantedBy=multi-user.target
```

Ative o serviço:

```bash
sudo systemctl daemon-reload
sudo systemctl enable flask_app
sudo systemctl start flask_app
sudo systemctl status flask_app
```

---

## 📊 Monitoramento

### MLflow Tracking

Acesse a interface do MLflow em: `http://localhost:8020`

**Recursos disponíveis:**
- Comparação de experimentos
- Visualização de métricas (MSE, RMSE, R²)
- Histórico de hiperparâmetros
- Versionamento de modelos
- Artifacts e logs

### Logs de Predição

Todas as predições são automaticamente salvas em `log_predictions.csv`:

```csv
timestamp,response_time,memory_usage
2025-02-06 10:30:00,0.0234,45.2
2025-02-06 10:31:15,0.0189,44.8
```

### Métricas em Tempo Real

O endpoint `/monitor` fornece:
- **CPU Usage**: Uso atual do processador
- **Memory Usage**: Consumo de memória RAM
- **Response Times**: Histórico de tempo de resposta
- **Request History**: Log completo de requisições

---

## 🧪 Testes

### Testar API Localmente

```bash
# Predição via curl
curl -X POST http://localhost:8010/predict \
  -H "Content-Type: application/json" \
  -d '{
    "predict": [23.30, 23.31, ..., 22.52]
  }'

# Monitoramento
curl http://localhost:8010/monitor
```

### Exemplo Python

```python
import requests
import json

# Dados de entrada (60 valores)
data = {
    "predict": [
        23.3007, 23.3082, 23.1652, 22.9243, 23.0824,
        # ... adicione os 55 valores restantes
    ]
}

# Realizar predição
response = requests.post(
    "http://localhost:8010/predict",
    json=data
)

result = response.json()
print(f"Preço previsto: ${result['predicted_price']}")
print(f"Tempo de resposta: {result['response_time_sec']}s")
```

---

## 🛠️ Tecnologias Utilizadas

### Core
- **Python 3.8+**: Linguagem principal
- **TensorFlow/Keras**: Framework de Deep Learning
- **NumPy**: Computação numérica
- **Pandas**: Manipulação de dados
- **Scikit-learn**: Pré-processamento e métricas

### API & Deploy
- **Flask**: Framework web
- **Flasgger**: Documentação Swagger/OpenAPI
- **Flask-Compress**: Compressão de resposta
- **Gunicorn**: WSGI HTTP Server (produção)
- **Docker**: Containerização

### ML Ops
- **MLflow**: Tracking de experimentos
- **yfinance**: Coleta de dados financeiros
- **Joblib**: Serialização de modelos

### Monitoramento
- **psutil**: Métricas de sistema
- **Pandas**: Logging estruturado

---

## 📚 Referências e Recursos

### Documentação Oficial
- [TensorFlow LSTM Guide](https://www.tensorflow.org/guide/keras/rnn)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)

### Artigos Relacionados
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Time Series Forecasting with Deep Learning](https://machinelearningmastery.com/time-series-forecasting-deep-learning/)

---

## 👥 Equipe

Desenvolvido como parte do **Tech Challenge - Fase 4**
**Pós-Tech FIAP - Machine Learning Engineering**

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:

1. Fazer um Fork do projeto
2. Criar uma branch para sua feature (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona NovaFeature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abrir um Pull Request

---

## 📞 Contato

Para dúvidas ou sugestões sobre o projeto:

- **Issues**: [GitHub Issues](https://github.com/seu-usuario/Tech-Challenge-F4/issues)
- **Discussões**: [GitHub Discussions](https://github.com/seu-usuario/Tech-Challenge-F4/discussions)

---

<div align="center">

**⭐ Se este projeto foi útil para você, considere dar uma estrela!**

Feito com ❤️ e ☕ para o Tech Challenge FIAP

</div>
