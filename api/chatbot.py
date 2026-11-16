"""
Chatbot do projeto RotaLivre.

Interpreta perguntas sobre acessibilidade urbana,
explica os modelos de Inteligência Artificial e
orienta o usuário sobre análises de locais e coordenadas.

As previsões dos modelos são experimentais e utilizam
uma base de dados sintética.
"""

import re
import unicodedata


def normalizar_texto(texto: str) -> str:
    """
    Padroniza o texto recebido pelo chatbot.

    Remove acentos, pontuação desnecessária e diferenças
    entre letras maiúsculas e minúsculas.
    """

    texto = texto.lower().strip()

    texto = unicodedata.normalize("NFD", texto)

    texto = "".join(
        caractere
        for caractere in texto
        if unicodedata.category(caractere) != "Mn"
    )

    # Substitui pontuações por espaços.
    texto = re.sub(r"[^\w\s]", " ", texto)

    # Remove espaços repetidos.
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def contem_alguma_palavra(texto: str, palavras: list[str]) -> bool:
    """
    Verifica se alguma expressão da lista está presente no texto.

    Utiliza limites de palavras para evitar correspondências
    acidentais dentro de outras palavras.
    """

    for palavra in palavras:
        expressao = r"\b" + re.escape(palavra) + r"\b"

        if re.search(expressao, texto):
            return True

    return False


def processar_mensagem(mensagem: str) -> dict[str, object]:
    """
    Identifica a intenção da mensagem e retorna uma resposta.

    O chatbot possui uma base de conhecimento sobre o RotaLivre
    e identifica solicitações de classificação e agrupamento.
    """

    if not isinstance(mensagem, str) or not mensagem.strip():
        return {
            "sucesso": False,
            "intencao": "mensagem_vazia",
            "resposta": "Por favor, escreva uma pergunta para começar."
        }

    texto = normalizar_texto(mensagem)

    # Identifica saudações.
    if texto in [
        "oi",
        "ola",
        "bom dia",
        "boa tarde",
        "boa noite",
        "e ai",
        "quem e voce",
        "quem e vc"
    ]:
        return {
            "sucesso": True,
            "intencao": "saudacao",
            "resposta": (
                "Olá! Sou o RotaLivre Assistant, um assistente "
                "sobre acessibilidade urbana.\n\n"
                "Posso explicar o projeto, apresentar os modelos "
                "de Inteligência Artificial e ajudar a consultar "
                "classificações de acessibilidade e agrupamentos "
                "geográficos."
            )
        }

    # Apresenta as funcionalidades do chatbot.
    if contem_alguma_palavra(texto, [
        "ajuda",
        "o que voce pode fazer",
        "como voce pode ajudar",
        "quais suas funcoes",
        "como usar o chatbot",
        "como utilizar o chatbot",
        "comandos disponiveis"
    ]):
        return {
            "sucesso": True,
            "intencao": "ajuda",
            "resposta": (
                "Posso ajudar você com o RotaLivre!\n\n"
                "Você pode perguntar:\n\n"
                "1. Sobre o projeto e seu funcionamento.\n"
                "2. Como funciona a classificação Random Forest.\n"
                "3. Como funciona o agrupamento geográfico KMeans.\n"
                "4. Quais informações são necessárias para analisar "
                "um local.\n"
                "5. Quais são as limitações das análises experimentais.\n\n"
                "Também posso encaminhar solicitações de classificação "
                "e agrupamento quando os dados necessários forem enviados."
            )
        }

    # Identifica perguntas sobre o objetivo do projeto.
    if contem_alguma_palavra(texto, [
        "o que e o rotalivre",
        "sobre o projeto",
        "objetivo do projeto",
        "como funciona o rotalivre",
        "para que serve o rotalivre",
        "qual a proposta do rotalivre"
    ]):
        return {
            "sucesso": True,
            "intencao": "sobre_projeto",
            "resposta": (
                "O RotaLivre é um projeto voltado à acessibilidade urbana. "
                "Sua proposta é ajudar na identificação e compreensão "
                "das condições de acessibilidade dos locais.\n\n"
                "A solução utiliza Inteligência Artificial para classificar "
                "condições de acessibilidade e identificar agrupamentos "
                "geográficos."
            )
        }

    # Explica as limitações dos modelos.
    if contem_alguma_palavra(texto, [
        "dados sinteticos",
        "dados reais",
        "limitacoes da ia",
        "limitacoes dos modelos",
        "previsao oficial",
        "avaliacao oficial",
        "resultado real",
        "hotspot real"
    ]):
        return {
            "sucesso": True,
            "intencao": "limitacoes",
            "resposta": (
                "As análises atuais do RotaLivre são experimentais "
                "e utilizam uma base de dados sintética.\n\n"
                "Isso significa que os resultados demonstram o "
                "funcionamento dos algoritmos, mas não representam "
                "uma avaliação oficial de acessibilidade nem a "
                "identificação comprovada de hotspots reais.\n\n"
                "Para obter resultados representativos das condições "
                "urbanas, é necessário utilizar dados reais, atualizados "
                "e devidamente validados."
            )
        }

    # Identifica solicitações de classificação de um local.
    # Essas expressões são verificadas antes das explicações gerais.
    if contem_alguma_palavra(texto, [
        "analise a acessibilidade",
        "analise acessibilidade",
        "analisar acessibilidade",
        "analisar local",
        "analise local",
        "avaliar local",
        "avalie este local",
        "classificar local",
        "classifique este local",
        "prever acessibilidade",
        "qual a condicao dessa calcada",
        "qual a condicao deste local",
        "condicao de acessibilidade",
        "verificar acessibilidade",
        "verifique a acessibilidade",
        "avaliar a calcada",
        "classificacao deste local",
        "quero analisar uma calcada",
        "quero avaliar uma calcada",
        "esse local e acessivel",
        "este local e acessivel"
    ]):
        return {
            "sucesso": True,
            "intencao": "solicitar_classificacao",
            "resposta": (
                "Posso realizar uma classificação experimental "
                "utilizando o modelo Random Forest!\n\n"
                "Para isso, preciso das seguintes informações:\n\n"
                "- Tipo de local;\n"
                "- Largura da calçada;\n"
                "- Inclinação;\n"
                "- Presença de rampa;\n"
                "- Presença de piso tátil;\n"
                "- Condição do piso;\n"
                "- Sinalização acessível;\n"
                "- Quantidade de relatos;\n"
                "- Quantidade de usuários distintos.\n\n"
                "A previsão será experimental e não substituirá "
                "uma avaliação presencial."
            )
        }

    # Identifica solicitações de agrupamento geográfico.
    if contem_alguma_palavra(texto, [
        "identifique a regiao",
        "identificar a regiao",
        "identifique a regiao deste local",
        "analisar coordenadas",
        "analisar a localizacao",
        "localizar cluster",
        "qual cluster",
        "em qual cluster",
        "qual agrupamento",
        "identificar agrupamento",
        "identifique o agrupamento",
        "agrupamento deste local",
        "regiao geografica",
        "onde esse local esta agrupado",
        "onde este local esta agrupado",
        "identificar regiao geografica",
        "analisar regiao deste local",
        "quero analisar as coordenadas"
    ]):
        return {
            "sucesso": True,
            "intencao": "solicitar_agrupamento",
            "resposta": (
                "Posso consultar o agrupamento geográfico experimental "
                "utilizando o modelo KMeans!\n\n"
                "Para isso, preciso das coordenadas do local:\n\n"
                "- Latitude;\n"
                "- Longitude.\n\n"
                "O resultado indicará a qual grupo geográfico "
                "experimental o local foi associado."
            )
        }

    # Orienta sobre os dados necessários para classificação.
    if contem_alguma_palavra(texto, [
        "dados para classificar",
        "informacoes para classificar",
        "o que preciso informar",
        "quais dados preciso",
        "dados necessarios",
        "informacoes necessarias para classificacao"
    ]):
        return {
            "sucesso": True,
            "intencao": "dados_necessarios",
            "resposta": (
                "Para realizar uma classificação com o Random Forest, "
                "são necessárias nove informações:\n\n"
                "1. Tipo de local;\n"
                "2. Largura da calçada em metros;\n"
                "3. Inclinação em graus;\n"
                "4. Presença de rampa;\n"
                "5. Presença de piso tátil;\n"
                "6. Condição do piso;\n"
                "7. Sinalização acessível;\n"
                "8. Quantidade de relatos;\n"
                "9. Quantidade de usuários distintos.\n\n"
                "Essas informações são enviadas à API para "
                "realizar a previsão experimental."
            )
        }

    # Explica as categorias de acessibilidade.
    if contem_alguma_palavra(texto, [
        "classificacao",
        "acessivel",
        "inacessivel",
        "parcial",
        "random forest"
    ]):
        return {
            "sucesso": True,
            "intencao": "classificacao",
            "resposta": (
                "O RotaLivre utiliza o algoritmo Random Forest para "
                "classificar experimentalmente as condições de "
                "acessibilidade em três categorias:\n\n"
                "ACESSIVEL: características associadas a melhores "
                "condições de acessibilidade.\n\n"
                "PARCIAL: características associadas a condições "
                "intermediárias.\n\n"
                "INACESSIVEL: características associadas a condições "
                "inadequadas de acessibilidade.\n\n"
                "As classificações são previsões experimentais e não "
                "substituem uma avaliação presencial."
            )
        }

    # Explica o agrupamento geográfico.
    if contem_alguma_palavra(texto, [
        "kmeans",
        "cluster",
        "agrupamento",
        "agrupamento geografico",
        "hotspot"
    ]):
        return {
            "sucesso": True,
            "intencao": "agrupamento",
            "resposta": (
                "O RotaLivre utiliza o algoritmo KMeans para agrupar "
                "locais com base em suas coordenadas geográficas "
                "(latitude e longitude).\n\n"
                "O modelo experimental foi configurado com três grupos. "
                "Esses agrupamentos ajudam a visualizar a distribuição "
                "geográfica dos pontos, mas não representam hotspots "
                "reais enquanto forem utilizados dados sintéticos."
            )
        }

    # Explica a utilização da Inteligência Artificial.
    if contem_alguma_palavra(texto, [
        "inteligencia artificial",
        "machine learning",
        "aprendizado de maquina",
        "modelos de ia",
        "quais modelos",
        "quais algoritmos"
    ]):
        return {
            "sucesso": True,
            "intencao": "inteligencia_artificial",
            "resposta": (
                "O RotaLivre utiliza dois modelos de Machine Learning:\n\n"
                "1. Random Forest: realiza a classificação das condições "
                "de acessibilidade.\n\n"
                "2. KMeans: realiza o agrupamento geográfico dos locais.\n\n"
                "Os modelos foram treinados experimentalmente com dados "
                "sintéticos."
            )
        }

    # Resposta para perguntas não identificadas.
    return {
        "sucesso": True,
        "intencao": "nao_identificada",
        "resposta": (
            "Ainda não consegui identificar sua pergunta.\n\n"
            "Você pode perguntar sobre:\n\n"
            "- O funcionamento do RotaLivre;\n"
            "- Acessibilidade urbana;\n"
            "- Classificação com Random Forest;\n"
            "- Agrupamento geográfico com KMeans;\n"
            "- Informações necessárias para realizar uma análise;\n"
            "- Limitações dos modelos experimentais.\n\n"
            "Se preferir, digite 'ajuda' para conhecer minhas funcionalidades."
        )
    }
