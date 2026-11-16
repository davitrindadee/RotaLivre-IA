# RotaLivre — Inteligência Artificial e Chatbot

Projeto desenvolvido para a Global Solution FIAP 2026/2, com foco em mobilidade urbana e acessibilidade.

A solução utiliza técnicas de Inteligência Artificial para analisar características de acessibilidade de locais urbanos e explorar sua distribuição geográfica. Também disponibiliza uma API Flask com endpoints para consulta dos modelos e interação com o Chatbot RotaLivre.

## Objetivo

Desenvolver uma solução experimental baseada em Inteligência Artificial para:

- Classificar características de acessibilidade de locais urbanos.
- Agrupar coordenadas geográficas utilizando aprendizado não supervisionado.
- Disponibilizar previsões por meio de uma API REST.
- Oferecer um chatbot capaz de orientar o usuário e coletar informações para executar as análises.

## Modelos de Inteligência Artificial

### 1. Classificação de acessibilidade — Random Forest

O modelo utiliza o algoritmo Random Forest para classificar locais em três categorias:

- `ACESSIVEL`
- `PARCIAL`
- `INACESSIVEL`

Características utilizadas:

- Tipo de local.
- Largura da calçada.
- Inclinação em graus.
- Presença de rampa.
- Presença de piso tátil.
- Regularidade do piso.
- Sinalização acessível.
- Quantidade de relatos.
- Quantidade de usuários distintos.

O modelo foi desenvolvido utilizando Scikit-learn, com preparação dos dados, treinamento, avaliação e exportação por Joblib.

**Resultados experimentais:**

| Métrica | Resultado |
|---|---:|
| Acurácia | 80,83% |
| Precisão | 77,97% |
| Recall | 82,50% |
| F1-score | 79,90% |

O conjunto de teste contém 120 registros. Os resultados representam somente o desempenho sobre dados sintéticos.

### 2. Agrupamento geográfico — KMeans

O segundo modelo utiliza KMeans para agrupar locais de acordo com suas coordenadas de latitude e longitude.

O objetivo é explorar a distribuição geográfica dos registros e identificar grupos espaciais.

Foram avaliadas diferentes quantidades de clusters utilizando:

- Método Elbow (inércia).
- Coeficiente Silhouette.
- Visualização dos grupos e centros aproximados.

A configuração experimental selecionada foi de 3 clusters, com coeficiente Silhouette de aproximadamente 0,8197.

Os agrupamentos são experimentais e não representam hotspots reais, pois a base utilizada é sintética.

## API Flask

A API disponibiliza os modelos e o chatbot por meio de endpoints REST.

### Endpoints

| Método | Endpoint | Descrição |
|---|---|---|
| GET | `/` | Informações da API |
| GET | `/health` | Verificação de funcionamento |
| GET | `/api/modelos` | Informações sobre os modelos disponíveis |
| POST | `/api/classificar` | Classificação de acessibilidade |
| POST | `/api/agrupar` | Agrupamento geográfico |
| POST | `/api/chatbot` | Interação com o chatbot |

### Exemplos de requisições

**Classificação de acessibilidade**

`POST /api/classificar`

```json
{
  "tipo_local": "calcada",
  "largura_calcada_m": 1.5,
  "inclinacao_graus": 5,
  "rampa": true,
  "piso_tatil": true,
  "piso_regular": true,
  "sinalizacao_acessivel": true,
  "quantidade_relatos": 10,
  "usuarios_distintos": 5
}
```

**Agrupamento geográfico**

`POST /api/agrupar`

```json
{
  "latitude": -23.5505,
  "longitude": -46.6333
}
```

**Interação com o chatbot**

`POST /api/chatbot`

```json
{
  "mensagem": "Identifique a região deste local"
}
```

O chatbot pode conduzir uma conversa interativa, solicitando latitude e longitude para executar o agrupamento ou coletando as informações necessárias para a classificação.

Nas etapas seguintes de uma conversa, deve ser enviado o `sessao_id` retornado pela API, junto com a mensagem do usuário.

## Estrutura do projeto

```text
RotaLivre-IA/
│
├── dados/
│   ├── locais_acessibilidade.csv
│   └── locais_com_clusters.csv
│
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb
│   ├── 02_modelo_classificacao.ipynb
│   └── 03_modelo_agrupamento.ipynb
│
├── modelos/
│   ├── modelo_classificacao.joblib
│   ├── metadata_classificacao.json
│   ├── modelo_agrupamento.joblib
│   └── metadata_agrupamento.json
│
├── api/
│   ├── app.py
│   └── chatbot.py
│
├── scripts/
│   └── gerar_dados_sinteticos.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Origem e limitações dos dados

A base utilizada nesta versão do projeto contém 600 registros sintéticos, gerados por um script Python.

Os registros simulados não representam locais reais nem medições oficiais de acessibilidade urbana.

As métricas dos modelos avaliam o comportamento experimental sobre essa base e não comprovam seu desempenho em ambientes reais.

A classificação atual não implementa integralmente a regra oficial de avaliação baseada na média ponderada dos relatos dos últimos 30 dias e na confirmação por pelo menos três usuários distintos.

O agrupamento geográfico também não identifica hotspots reais, pois utiliza coordenadas simuladas.

## Tecnologias utilizadas

- Python 3.13
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Flask
- Joblib
- REST API

## Instalação e execução local

### 1. Clonar o repositório

```bash
git clone URL_DO_REPOSITORIO
```

Acesse a pasta do projeto:

```bash
cd RotaLivre-IA
```

### 2. Criar e ativar o ambiente virtual

```bash
python -m venv .venv
```

No Windows:

```bash
.venv\Scripts\activate
```

### 3. Instalar as dependências

```bash
python -m pip install -r requirements.txt
```

### 4. Gerar a base de dados

```bash
python scripts/gerar_dados_sinteticos.py
```

### 5. Executar os notebooks

Abra os arquivos da pasta `notebooks` e execute as células em ordem, caso queira reproduzir a análise e o treinamento dos modelos.

Os arquivos `.joblib` exportados devem estar disponíveis na pasta `modelos` para a execução da API.

### 6. Iniciar a API

Na raiz do projeto:

```bash
python api/app.py
```

A API estará disponível localmente em:

```text
http://127.0.0.1:5000
```

Para verificar o funcionamento:

```text
http://127.0.0.1:5000/health
```

## Publicação da API

A API pode ser publicada em uma plataforma de hospedagem compatível com aplicações Python, como o Render.

Após a publicação, os endpoints poderão ser acessados por uma URL pública, permitindo a integração com o Front-end RotaLivre.

**URL pública da API:** A preencher após a publicação.

## Integrantes

Projeto desenvolvido para a Global Solution FIAP 2026/2.

## Considerações finais

O RotaLivre busca explorar o uso da Inteligência Artificial como ferramenta de apoio à análise de acessibilidade urbana, combinando classificação, agrupamento geográfico e interação por chatbot.

Os resultados apresentados nesta versão são experimentais e dependem de validação futura com dados reais.