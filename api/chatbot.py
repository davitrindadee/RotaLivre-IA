"""
Chatbot do projeto RotaLivre.

Este módulo utiliza regras e uma base de conhecimento
para responder perguntas sobre acessibilidade urbana
e os modelos de Inteligência Artificial do projeto.
"""

import re
import unicodedata


def normalizar_texto(texto: str) -> str:
    """
    Padroniza o texto recebido.

    Remove acentos, espaços extras e transforma
    as letras em minúsculas.
    """
    texto = texto.lower().strip()

    texto = unicodedata.normalize("NFD", texto)

    texto = "".join(
        caractere
        for caractere in texto
        if unicodedata.category(caractere) != "Mn"
    )

    texto = re.sub(r"\s+", " ", texto)

    return texto


def contem_termo(texto: str, termos: list[str]) -> bool:
    """
    Verifica se o texto contém alguma palavra ou expressão.

    Utiliza limites de palavras para evitar identificações
    incorretas dentro de outras palavras.
    """
    for termo in termos:
        padrao = rf"\b{re.escape(termo)}\b"

        if re.search(padrao, texto):
            return True

    return False


def processar_mensagem(mensagem: str) -> dict[str, str]:
    """
    Identifica a intenção da mensagem e prepara uma resposta.

    O chatbot utiliza regras e respostas previamente definidas.
    A consulta aos modelos é realizada pela API Flask.
    """

    if not isinstance(mensagem, str) or not mensagem.strip():
        return {
            "sucesso": False,
            "intencao": "mensagem_vazia",
            "resposta": "Por favor, escreva uma pergunta para começar."
        }

    texto = normalizar_texto(mensagem)

    # Identifica saudações.
    if contem_termo(texto, [
        "oi",
        "ola",
        "bom dia",
        "boa tarde",
        "boa noite",
        "quem e voce",
        "quem e vc"
    ]):
        return {
            "sucesso": True,
            "intencao": "saudacao",
            "resposta": (
                "Olá! Sou o RotaLivre Assistant, "
                "um assistente sobre acessibilidade urbana. "
                "Posso explicar o projeto, as classificações "
                "de acessibilidade e os modelos de Inteligência Artificial."
            )
        }

    # Identifica pedidos de classificação de um local.
    if contem_termo(texto, [
        "analise a acessibilidade",
        "analisar acessibilidade",
        "analisar local",
        "analisar este local",
        "avaliar local",
        "avaliar este local",
        "classificar local",
        "classificar este local",
        "prever acessibilidade",
        "quero classificar",
        "quero uma classificacao",
        "fazer uma classificacao"
    ]):
        return {
            "sucesso": True,
            "intencao": "solicitar_classificacao",
            "resposta": (
                "Posso ajudar com uma classificação experimental! "
                "O modelo Random Forest precisa das informações do local: "
                "tipo de local, largura da calçada, inclinação, presença "
                "de rampa, piso tátil, piso regular, sinalização acessível, "
                "quantidade de relatos e usuários distintos. "
                "Esses dados podem ser enviados à API no campo dados_local."
            )
        }

    # Identifica pedidos de agrupamento geográfico.
    if contem_termo(texto, [
        "analisar coordenadas",
        "identifique a regiao",
        "identificar regiao",
        "qual cluster",
        "qual e o cluster",
        "localizar cluster",
        "consultar cluster",
        "agrupar coordenadas",
        "identificar agrupamento",
        "analisar regiao geografica"
    ]):
        return {
            "sucesso": True,
            "intencao": "solicitar_agrupamento",
            "resposta": (
                "Posso consultar o agrupamento geográfico experimental! "
                "Preciso das coordenadas do local: latitude e longitude. "
                "Elas podem ser enviadas à API no campo coordenadas."
            )
        }

    # Explica a origem dos dados utilizados.
    if contem_termo(texto, [
        "dados sinteticos",
        "dados reais",
        "base real",
        "origem dos dados",
        "dados utilizados",
        "dataset",
        "os dados sao reais",
        "os dados sao sinteticos"
    ]):
        return {
            "sucesso": True,
            "intencao": "origem_dados",
            "resposta": (
                "Os modelos atuais do RotaLivre foram treinados "
                "experimentalmente com uma base de dados sintéticos "
                "e simulados. Portanto, as previsões de acessibilidade "
                "e os agrupamentos geográficos não representam "
                "avaliações reais nem hotspots confirmados."
            )
        }

    # Explica as categorias de acessibilidade.
    if contem_termo(texto, [
        "classificacao",
        "acessivel",
        "inacessivel",
        "parcial",
        "random forest",
        "categorias",
        "classe de acessibilidade"
    ]):
        return {
            "sucesso": True,
            "intencao": "classificacao",
            "resposta": (
                "O RotaLivre utiliza o algoritmo Random Forest "
                "para classificar experimentalmente as condições "
                "de acessibilidade em três categorias:\n\n"
                "ACESSIVEL: características associadas a melhores "
                "condições de acessibilidade.\n\n"
                "PARCIAL: características associadas a condições "
                "intermediárias.\n\n"
                "INACESSIVEL: características associadas a condições "
                "inadequadas de acessibilidade.\n\n"
                "A classificação é uma previsão do modelo, não uma "
                "avaliação oficial ou presencial."
            )
        }

    # Explica o agrupamento geográfico.
    if contem_termo(texto, [
        "kmeans",
        "cluster",
        "agrupamento",
        "hotspot",
        "latitude",
        "longitude",
        "coordenadas"
    ]):
        return {
            "sucesso": True,
            "intencao": "agrupamento",
            "resposta": (
                "O RotaLivre utiliza o algoritmo KMeans para agrupar "
                "locais com base em suas coordenadas geográficas, "
                "latitude e longitude.\n\n"
                "O modelo experimental foi configurado com três grupos. "
                "Os agrupamentos ajudam a visualizar a distribuição "
                "geográfica dos pontos, mas não representam hotspots "
                "reais enquanto forem utilizados dados sintéticos."
            )
        }

    # Explica acessibilidade urbana.
    if contem_termo(texto, [
        "acessibilidade urbana",
        "acessibilidade",
        "calcada",
        "rampa",
        "piso tatil",
        "mobilidade urbana"
    ]):
        return {
            "sucesso": True,
            "intencao": "acessibilidade_urbana",
            "resposta": (
                "A acessibilidade urbana está relacionada às condições "
                "que permitem que diferentes pessoas utilizem espaços "
                "públicos com segurança e autonomia.\n\n"
                "Entre os aspectos importantes estão largura e inclinação "
                "das calçadas, rampas, piso tátil, regularidade do piso "
                "e sinalização acessível.\n\n"
                "O RotaLivre utiliza esses aspectos como parte das "
                "informações analisadas pelo seu modelo de classificação."
            )
        }

    # Explica os modelos de Inteligência Artificial.
    if contem_termo(texto, [
        "inteligencia artificial",
        "machine learning",
        "aprendizado de maquina",
        "modelos de ia",
        "modelos utilizados"
    ]):
        return {
            "sucesso": True,
            "intencao": "inteligencia_artificial",
            "resposta": (
                "O RotaLivre utiliza dois modelos de Machine Learning:\n\n"
                "1. Random Forest: realiza a classificação experimental "
                "das condições de acessibilidade.\n\n"
                "2. KMeans: realiza o agrupamento geográfico dos locais.\n\n"
                "Os modelos foram treinados com dados sintéticos. "
                "O chatbot, nesta versão, utiliza regras e uma base "
                "de conhecimento para interpretar perguntas."
            )
        }

    # Explica o funcionamento geral do projeto.
    if contem_termo(texto, [
        "o que e o rotalivre",
        "sobre o projeto",
        "objetivo do projeto",
        "como funciona o rotalivre",
        "qual a finalidade do projeto"
    ]):
        return {
            "sucesso": True,
            "intencao": "sobre_projeto",
            "resposta": (
                "O RotaLivre é um projeto voltado à acessibilidade urbana. "
                "Sua proposta é apoiar a identificação e compreensão "
                "das condições de acessibilidade dos locais.\n\n"
                "A solução conta com modelos de Inteligência Artificial "
                "para classificação de acessibilidade e agrupamento "
                "geográfico, além de um chatbot para responder dúvidas "
                "sobre o projeto."
            )
        }

    # Apresenta as funcionalidades disponíveis.
    if contem_termo(texto, [
        "ajuda",
        "o que voce faz",
        "quais suas funcoes",
        "como posso usar"
    ]):
        return {
            "sucesso": True,
            "intencao": "ajuda",
            "resposta": (
                "Posso ajudar com os seguintes assuntos:\n\n"
                "- Informações sobre o projeto RotaLivre.\n"
                "- Acessibilidade urbana.\n"
                "- Categorias ACESSIVEL, PARCIAL e INACESSIVEL.\n"
                "- Modelo Random Forest.\n"
                "- Algoritmo KMeans e agrupamento geográfico.\n"
                "- Origem dos dados utilizados.\n\n"
                "Também posso acompanhar análises experimentais "
                "quando os dados estruturados são enviados à API."
            )
        }

    # Resposta padrão para perguntas não identificadas.
    return {
        "sucesso": True,
        "intencao": "nao_identificada",
        "resposta": (
            "Ainda não consegui identificar sua pergunta. "
            "Você pode perguntar sobre o RotaLivre, acessibilidade urbana, "
            "classificação Random Forest, agrupamento KMeans ou "
            "origem dos dados utilizados."
        )
    }
