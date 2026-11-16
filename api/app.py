from pathlib import Path
import math
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request

# Importa o chatbot, tanto na execucao direta quanto por modulo.
try:
    from .chatbot import processar_mensagem
except ImportError:
    from chatbot import processar_mensagem


app = Flask(__name__)

# Caminhos dos arquivos do projeto.
RAIZ = Path(__file__).resolve().parent.parent
PASTA_MODELOS = RAIZ / "modelos"

ARQUIVO_CLASSIFICACAO = PASTA_MODELOS / "modelo_classificacao.joblib"
ARQUIVO_AGRUPAMENTO = PASTA_MODELOS / "modelo_agrupamento.joblib"

COLUNAS = [
    "tipo_local",
    "largura_calcada_m",
    "inclinacao_graus",
    "rampa",
    "piso_tatil",
    "piso_regular",
    "sinalizacao_acessivel",
    "quantidade_relatos",
    "usuarios_distintos"
]

CAMPOS_NUMERICOS = [
    "largura_calcada_m",
    "inclinacao_graus",
    "quantidade_relatos",
    "usuarios_distintos"
]

CAMPOS_BOOLEANOS = [
    "rampa",
    "piso_tatil",
    "piso_regular",
    "sinalizacao_acessivel"
]

# Armazena temporariamente as conversas dos usuarios.
# Cada sessao possui sua propria intencao, etapa e dados.
sessoes_chatbot = {}

# Perguntas utilizadas na classificacao com Random Forest.
PERGUNTAS_CLASSIFICACAO = [
    {
        "campo": "tipo_local",
        "pergunta": "Qual e o tipo de local? Exemplo: calcada."
    },
    {
        "campo": "largura_calcada_m",
        "pergunta": "Qual e a largura da calcada em metros? Exemplo: 1.5."
    },
    {
        "campo": "inclinacao_graus",
        "pergunta": "Qual e a inclinacao em graus? Exemplo: 5."
    },
    {
        "campo": "rampa",
        "pergunta": "O local possui rampa? Responda sim ou nao."
    },
    {
        "campo": "piso_tatil",
        "pergunta": "O local possui piso tatil? Responda sim ou nao."
    },
    {
        "campo": "piso_regular",
        "pergunta": "O piso esta em boas condicoes e e regular? Responda sim ou nao."
    },
    {
        "campo": "sinalizacao_acessivel",
        "pergunta": "O local possui sinalizacao acessivel? Responda sim ou nao."
    },
    {
        "campo": "quantidade_relatos",
        "pergunta": "Quantos relatos existem sobre esse local?"
    },
    {
        "campo": "usuarios_distintos",
        "pergunta": "Quantos usuarios distintos fizeram relatos?"
    }
]

# Perguntas utilizadas no agrupamento geografico com KMeans.
PERGUNTAS_AGRUPAMENTO = [
    {
        "campo": "latitude",
        "pergunta": (
            "Qual e a latitude do local? "
            "Exemplo: -23.5505."
        )
    },
    {
        "campo": "longitude",
        "pergunta": (
            "Qual e a longitude do local? "
            "Exemplo: -46.6333."
        )
    }
]


# Carrega os dois modelos utilizados pela API.
def carregar_modelos():
    if not ARQUIVO_CLASSIFICACAO.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {ARQUIVO_CLASSIFICACAO}"
        )

    if not ARQUIVO_AGRUPAMENTO.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {ARQUIVO_AGRUPAMENTO}"
        )

    try:
        classificacao = joblib.load(ARQUIVO_CLASSIFICACAO)
        agrupamento = joblib.load(ARQUIVO_AGRUPAMENTO)

    except EOFError as erro:
        raise RuntimeError(
            "Um dos modelos esta incompleto. "
            "Execute novamente as celulas de exportacao dos notebooks."
        ) from erro

    return classificacao, agrupamento


modelo_classificacao, pacote_agrupamento = carregar_modelos()
modelo_kmeans = pacote_agrupamento["modelo"]


# Permite o acesso a API por aplicacoes externas.
@app.after_request
def configurar_cors(resposta):
    resposta.headers["Access-Control-Allow-Origin"] = "*"
    resposta.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resposta.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resposta


# Valida os dados enviados para a classificacao.
def validar_dados_classificacao(dados):
    if not isinstance(dados, dict):
        return None, "Envie os dados do local em formato JSON."

    campos_ausentes = [
        campo for campo in COLUNAS if campo not in dados
    ]

    if campos_ausentes:
        return None, (
            "Preencha todos os campos obrigatorios. "
            f"Campos ausentes: {', '.join(campos_ausentes)}"
        )

    if (
        not isinstance(dados["tipo_local"], str)
        or not dados["tipo_local"].strip()
    ):
        return None, "Informe um tipo de local valido."

    entrada = dados.copy()

    for campo in CAMPOS_BOOLEANOS:
        if not isinstance(entrada[campo], bool):
            return None, f"O campo {campo} deve ser true ou false."

    try:
        for campo in CAMPOS_NUMERICOS:
            if isinstance(entrada[campo], bool):
                raise ValueError

            entrada[campo] = float(entrada[campo])

            if not math.isfinite(entrada[campo]):
                raise ValueError

    except (ValueError, TypeError):
        return None, "Os campos numericos devem conter valores validos."

    if entrada["largura_calcada_m"] < 0:
        return None, "A largura nao pode ser negativa."

    if entrada["inclinacao_graus"] < 0:
        return None, "A inclinacao nao pode ser negativa."

    if (
        entrada["quantidade_relatos"] < 0
        or entrada["usuarios_distintos"] < 0
    ):
        return None, "As quantidades nao podem ser negativas."

    return entrada, None


# Executa a previsao do modelo Random Forest.
def executar_classificacao(entrada):
    dados_local = pd.DataFrame(
        [[entrada[campo] for campo in COLUNAS]],
        columns=COLUNAS
    )

    previsao = modelo_classificacao.predict(dados_local)[0]

    return str(previsao)


# Valida as coordenadas geograficas recebidas.
def validar_coordenadas(dados):
    if not isinstance(dados, dict):
        return None, None, "Envie as coordenadas em formato JSON."

    if "latitude" not in dados or "longitude" not in dados:
        return None, None, "Informe latitude e longitude."

    try:
        if (
            isinstance(dados["latitude"], bool)
            or isinstance(dados["longitude"], bool)
        ):
            raise ValueError

        latitude = float(dados["latitude"])
        longitude = float(dados["longitude"])

        if not math.isfinite(latitude) or not math.isfinite(longitude):
            raise ValueError

    except (ValueError, TypeError):
        return None, None, (
            "Latitude e longitude devem ser numeros validos."
        )

    if not -90 <= latitude <= 90:
        return None, None, "Latitude invalida."

    if not -180 <= longitude <= 180:
        return None, None, "Longitude invalida."

    return latitude, longitude, None


# Executa o agrupamento geografico com o modelo KMeans.
def executar_agrupamento(latitude, longitude):
    lat_referencia = pacote_agrupamento["latitude_referencia"]
    lon_referencia = pacote_agrupamento["longitude_referencia"]
    fator_lon = pacote_agrupamento["fator_longitude"]
    fator_lat = pacote_agrupamento["fator_latitude"]

    # Converte as coordenadas para quilometros.
    x = (longitude - lon_referencia) * fator_lon
    y = (latitude - lat_referencia) * fator_lat

    coordenadas = pd.DataFrame(
        [[x, y]],
        columns=["x_km", "y_km"]
    )

    cluster = int(modelo_kmeans.predict(coordenadas)[0])

    centro_x, centro_y = modelo_kmeans.cluster_centers_[cluster]

    centro_latitude = (centro_y / fator_lat) + lat_referencia
    centro_longitude = (centro_x / fator_lon) + lon_referencia

    distancia = math.hypot(x - centro_x, y - centro_y)

    return {
        "cluster": cluster,
        "centro_aproximado": {
            "latitude": round(float(centro_latitude), 6),
            "longitude": round(float(centro_longitude), 6)
        },
        "distancia_ate_centro_km": round(distancia, 3)
    }


# Converte respostas simples em valores booleanos.
def interpretar_booleano(texto):
    texto = texto.strip().lower()

    respostas_sim = [
        "sim", "s", "true", "1", "possui",
        "tem", "positivo", "presente"
    ]

    respostas_nao = [
        "nao", "não", "n", "false", "0",
        "nao possui", "não possui", "ausente"
    ]

    if texto in respostas_sim:
        return True

    if texto in respostas_nao:
        return False

    return None


# Interpreta a resposta de acordo com o campo solicitado.
def interpretar_resposta(campo, mensagem):
    mensagem = mensagem.strip()

    if campo == "tipo_local":
        if not mensagem:
            return None, "Informe o tipo de local."

        return mensagem.lower(), None

    if campo in CAMPOS_BOOLEANOS:
        valor = interpretar_booleano(mensagem)

        if valor is None:
            return None, "Responda apenas sim ou nao."

        return valor, None

    if campo in CAMPOS_NUMERICOS:
        try:
            valor = float(mensagem.replace(",", "."))

            if not math.isfinite(valor) or valor < 0:
                raise ValueError

            return valor, None

        except ValueError:
            return None, "Informe um numero valido e nao negativo."

    # Valida as respostas de latitude e longitude.
    if campo in ["latitude", "longitude"]:
        try:
            valor = float(mensagem.replace(",", "."))

            if not math.isfinite(valor):
                raise ValueError

            if campo == "latitude" and not -90 <= valor <= 90:
                return None, "A latitude deve estar entre -90 e 90."

            if campo == "longitude" and not -180 <= valor <= 180:
                return None, "A longitude deve estar entre -180 e 180."

            return valor, None

        except ValueError:
            return None, "Informe uma coordenada numerica valida."

    return None, "Nao foi possivel interpretar essa resposta."


# Inicia uma nova conversa de classificacao.
def iniciar_classificacao():
    sessao_id = str(uuid.uuid4())

    sessoes_chatbot[sessao_id] = {
        "intencao": "solicitar_classificacao",
        "etapa": 0,
        "dados": {}
    }

    return sessao_id


# Inicia uma nova conversa de agrupamento geografico.
def iniciar_agrupamento():
    sessao_id = str(uuid.uuid4())

    sessoes_chatbot[sessao_id] = {
        "intencao": "solicitar_agrupamento",
        "etapa": 0,
        "dados": {}
    }

    return sessao_id


# Processa as respostas da conversa de classificacao.
def continuar_classificacao(sessao_id, mensagem):
    sessao = sessoes_chatbot[sessao_id]
    etapa = sessao["etapa"]

    campo_atual = PERGUNTAS_CLASSIFICACAO[etapa]["campo"]

    # Permite cancelar a coleta de informacoes.
    if mensagem.strip().lower() in [
        "cancelar", "cancelar analise", "sair"
    ]:
        del sessoes_chatbot[sessao_id]

        return {
            "sucesso": True,
            "intencao": "classificacao_cancelada",
            "resposta": (
                "A analise foi cancelada. "
                "Quando quiser, podemos comecar novamente."
            )
        }

    valor, erro = interpretar_resposta(campo_atual, mensagem)

    if erro:
        return {
            "sucesso": False,
            "intencao": "solicitar_classificacao",
            "sessao_id": sessao_id,
            "campo_solicitado": campo_atual,
            "resposta": (
                f"{erro}\n\n"
                f"{PERGUNTAS_CLASSIFICACAO[etapa]['pergunta']}"
            )
        }

    # Guarda a resposta na sessao atual.
    sessao["dados"][campo_atual] = valor
    sessao["etapa"] += 1

    # Verifica se ainda existem perguntas.
    if sessao["etapa"] < len(PERGUNTAS_CLASSIFICACAO):
        proxima = PERGUNTAS_CLASSIFICACAO[sessao["etapa"]]

        return {
            "sucesso": True,
            "intencao": "coletando_classificacao",
            "sessao_id": sessao_id,
            "etapa_atual": sessao["etapa"] + 1,
            "total_etapas": len(PERGUNTAS_CLASSIFICACAO),
            "campo_solicitado": proxima["campo"],
            "resposta": proxima["pergunta"]
        }

    # Quando todas as respostas chegam, executa o Random Forest.
    entrada, erro = validar_dados_classificacao(sessao["dados"])

    if erro:
        del sessoes_chatbot[sessao_id]

        return {
            "sucesso": False,
            "intencao": "erro_classificacao",
            "resposta": erro
        }

    previsao = executar_classificacao(entrada)

    del sessoes_chatbot[sessao_id]

    return {
        "sucesso": True,
        "intencao": "classificacao_concluida",
        "analise_classificacao": {
            "algoritmo": "Random Forest",
            "situacao_acessibilidade": previsao,
            "observacao": (
                "Previsao experimental baseada em dados sinteticos. "
                "Nao representa uma avaliacao oficial."
            )
        },
        "resposta": (
            f"Conclui a analise experimental!\n\n"
            f"Situacao de acessibilidade prevista: {previsao}.\n\n"
            "Lembre-se: o resultado utiliza dados sinteticos e "
            "nao substitui uma avaliacao presencial."
        )
    }


# Processa as respostas da conversa de agrupamento geografico.
def continuar_agrupamento(sessao_id, mensagem):
    sessao = sessoes_chatbot[sessao_id]
    etapa = sessao["etapa"]

    campo_atual = PERGUNTAS_AGRUPAMENTO[etapa]["campo"]

    # Permite cancelar a coleta de coordenadas.
    if mensagem.strip().lower() in [
        "cancelar", "cancelar analise", "sair"
    ]:
        del sessoes_chatbot[sessao_id]

        return {
            "sucesso": True,
            "intencao": "agrupamento_cancelado",
            "resposta": (
                "A consulta geografica foi cancelada. "
                "Quando quiser, podemos comecar novamente."
            )
        }

    valor, erro = interpretar_resposta(campo_atual, mensagem)

    if erro:
        return {
            "sucesso": False,
            "intencao": "solicitar_agrupamento",
            "sessao_id": sessao_id,
            "campo_solicitado": campo_atual,
            "resposta": (
                f"{erro}\n\n"
                f"{PERGUNTAS_AGRUPAMENTO[etapa]['pergunta']}"
            )
        }

    # Guarda a coordenada informada pelo usuario.
    sessao["dados"][campo_atual] = valor
    sessao["etapa"] += 1

    # Solicita a proxima coordenada, se ainda estiver faltando.
    if sessao["etapa"] < len(PERGUNTAS_AGRUPAMENTO):
        proxima = PERGUNTAS_AGRUPAMENTO[sessao["etapa"]]

        return {
            "sucesso": True,
            "intencao": "coletando_agrupamento",
            "sessao_id": sessao_id,
            "etapa_atual": sessao["etapa"] + 1,
            "total_etapas": len(PERGUNTAS_AGRUPAMENTO),
            "campo_solicitado": proxima["campo"],
            "resposta": proxima["pergunta"]
        }

    # Valida as duas coordenadas antes de executar o KMeans.
    latitude, longitude, erro = validar_coordenadas(
        sessao["dados"]
    )

    if erro:
        del sessoes_chatbot[sessao_id]

        return {
            "sucesso": False,
            "intencao": "erro_agrupamento",
            "resposta": erro
        }

    # Executa o modelo KMeans com as coordenadas coletadas.
    resultado = executar_agrupamento(latitude, longitude)

    del sessoes_chatbot[sessao_id]

    return {
        "sucesso": True,
        "intencao": "agrupamento_concluido",
        "analise_geografica": {
            "algoritmo": "KMeans",
            **resultado,
            "observacao": (
                "Agrupamento experimental baseado em coordenadas "
                "sinteticas. Nao representa um hotspot real."
            )
        },
        "resposta": (
            "Conclui a analise geografica experimental!\n\n"
            f"O local foi associado ao cluster {resultado['cluster']}.\n\n"
            f"Centro aproximado: "
            f"latitude {resultado['centro_aproximado']['latitude']} e "
            f"longitude {resultado['centro_aproximado']['longitude']}.\n\n"
            f"Distancia ate o centro do grupo: "
            f"{resultado['distancia_ate_centro_km']} km.\n\n"
            "Lembre-se: o agrupamento utiliza dados sinteticos "
            "e nao representa um hotspot real."
        )
    }


# Direciona cada resposta para o fluxo correspondente.
def continuar_conversa(sessao_id, mensagem):
    sessao = sessoes_chatbot[sessao_id]

    if sessao["intencao"] == "solicitar_classificacao":
        return continuar_classificacao(sessao_id, mensagem)

    if sessao["intencao"] == "solicitar_agrupamento":
        return continuar_agrupamento(sessao_id, mensagem)

    return {
        "sucesso": False,
        "erro": "Tipo de conversa nao identificado."
    }


@app.route("/", methods=["GET"])
def inicio():
    return jsonify({
        "projeto": "RotaLivre",
        "mensagem": "API de Inteligencia Artificial funcionando!",
        "versao": "1.3.0"
    })


@app.route("/health", methods=["GET"])
def verificar_status():
    return jsonify({
        "status": "online",
        "modelo_classificacao": "carregado",
        "modelo_agrupamento": "carregado",
        "chatbot": "carregado",
        "conversa_interativa": "disponivel"
    })


@app.route("/api/modelos", methods=["GET"])
def listar_modelos():
    return jsonify({
        "modelos": [
            {
                "nome": "Classificacao de acessibilidade",
                "algoritmo": "Random Forest",
                "endpoint": "/api/classificar",
                "metodo": "POST"
            },
            {
                "nome": "Agrupamento geografico",
                "algoritmo": "KMeans",
                "endpoint": "/api/agrupar",
                "metodo": "POST"
            },
            {
                "nome": "Assistente virtual RotaLivre",
                "algoritmo": "Chatbot com integracao aos modelos de IA",
                "endpoint": "/api/chatbot",
                "metodo": "POST"
            }
        ]
    })


@app.route("/api/classificar", methods=["POST"])
def classificar_local():
    dados = request.get_json(silent=True)

    if not isinstance(dados, dict):
        return jsonify({"erro": "Envie um JSON valido."}), 400

    entrada, erro = validar_dados_classificacao(dados)

    if erro:
        return jsonify({"erro": erro}), 400

    previsao = executar_classificacao(entrada)

    return jsonify({
        "sucesso": True,
        "situacao_acessibilidade": previsao,
        "observacao": (
            "Previsao experimental baseada em dados sinteticos. "
            "Nao representa uma avaliacao oficial de um local real."
        )
    })


@app.route("/api/agrupar", methods=["POST"])
def agrupar_local():
    dados = request.get_json(silent=True)

    if not isinstance(dados, dict):
        return jsonify({"erro": "Envie um JSON valido."}), 400

    latitude, longitude, erro = validar_coordenadas(dados)

    if erro:
        return jsonify({"erro": erro}), 400

    resultado = executar_agrupamento(latitude, longitude)

    return jsonify({
        "sucesso": True,
        **resultado,
        "observacao": (
            "Agrupamento experimental baseado em coordenadas sinteticas. "
            "Nao representa um hotspot real."
        )
    })


# Endpoint do chatbot integrado aos dois modelos de IA.
@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    dados = request.get_json(silent=True)

    if not isinstance(dados, dict):
        return jsonify({"erro": "Envie um JSON valido."}), 400

    mensagem = dados.get("mensagem")
    sessao_id = dados.get("sessao_id")

    if not isinstance(mensagem, str) or not mensagem.strip():
        return jsonify({
            "sucesso": False,
            "erro": "Envie uma mensagem valida para o chatbot."
        }), 400

    mensagem = mensagem.strip()

    # Continua uma conversa que ja possui uma sessao ativa.
    if sessao_id is not None:
        if (
            not isinstance(sessao_id, str)
            or sessao_id not in sessoes_chatbot
        ):
            return jsonify({
                "sucesso": False,
                "erro": "Sessao nao encontrada. Inicie uma nova conversa."
            }), 404

        resultado_conversa = continuar_conversa(
            sessao_id,
            mensagem
        )

        return jsonify(resultado_conversa)

    # Permite enviar os dados completos para classificacao diretamente.
    if "dados_local" in dados:
        entrada, erro = validar_dados_classificacao(
            dados["dados_local"]
        )

        if erro:
            return jsonify({"erro": erro}), 400

        previsao = executar_classificacao(entrada)

        return jsonify({
            "sucesso": True,
            "intencao": "solicitar_classificacao",
            "analise_classificacao": {
                "algoritmo": "Random Forest",
                "situacao_acessibilidade": previsao,
                "observacao": (
                    "Previsao experimental baseada em dados sinteticos. "
                    "Nao representa uma avaliacao oficial."
                )
            },
            "resposta": (
                f"Analise experimental do local: {previsao}."
            )
        })

    # Permite enviar coordenadas completas para o KMeans diretamente.
    if "coordenadas" in dados:
        latitude, longitude, erro = validar_coordenadas(
            dados["coordenadas"]
        )

        if erro:
            return jsonify({"erro": erro}), 400

        resultado_cluster = executar_agrupamento(
            latitude,
            longitude
        )

        return jsonify({
            "sucesso": True,
            "intencao": "solicitar_agrupamento",
            "analise_geografica": {
                "algoritmo": "KMeans",
                **resultado_cluster,
                "observacao": (
                    "Agrupamento experimental baseado em coordenadas "
                    "sinteticas. Nao representa um hotspot real."
                )
            },
            "resposta": (
                f"O local foi associado ao cluster "
                f"{resultado_cluster['cluster']} no agrupamento experimental."
            )
        })

    # Identifica a intencao da mensagem do usuario.
    resultado = processar_mensagem(mensagem)

    resposta = {
        "sucesso": resultado["sucesso"],
        "intencao": resultado["intencao"],
        "resposta": resultado["resposta"]
    }

    # Inicia a coleta de dados para classificacao.
    if resultado["intencao"] == "solicitar_classificacao":
        novo_id = iniciar_classificacao()

        resposta["sessao_id"] = novo_id
        resposta["etapa_atual"] = 1
        resposta["total_etapas"] = len(PERGUNTAS_CLASSIFICACAO)
        resposta["campo_solicitado"] = "tipo_local"
        resposta["resposta"] = PERGUNTAS_CLASSIFICACAO[0]["pergunta"]

    # Inicia a coleta de coordenadas para agrupamento geografico.
    elif resultado["intencao"] == "solicitar_agrupamento":
        novo_id = iniciar_agrupamento()

        resposta["sessao_id"] = novo_id
        resposta["etapa_atual"] = 1
        resposta["total_etapas"] = len(PERGUNTAS_AGRUPAMENTO)
        resposta["campo_solicitado"] = "latitude"
        resposta["resposta"] = PERGUNTAS_AGRUPAMENTO[0]["pergunta"]

    return jsonify(resposta)


@app.errorhandler(404)
def rota_nao_encontrada(erro):
    return jsonify({
        "erro": "Endpoint nao encontrado."
    }), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
