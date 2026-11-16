"""
Gera uma base demonstrativa de locais urbanos para o projeto RotaLivre.

"""

from pathlib import Path

import numpy as np
import pandas as pd


# Quantidade de locais simulados para o protótipo.
QUANTIDADE_LOCAIS = 600

# Valores possíveis para os tipos de locais do RotaLivre.
TIPOS_LOCAIS = [
    "CALCADA",
    "RAMPA",
    "ELEVADOR",
    "ESTABELECIMENTO",
    "OBSTACULO",
]

# Centro aproximado de São Paulo para gerar coordenadas demonstrativas.
CENTRO_LATITUDE = -23.5505
CENTRO_LONGITUDE = -46.6333

# Garante que a geração possa ser reproduzida.
np.random.seed(42)


def classificar_acessibilidade(nota):
    """Classifica a acessibilidade conforme os limites do Front-end."""
    if nota >= 70:
        return "ACESSIVEL"

    if nota >= 40:
        return "PARCIAL"

    return "INACESSIVEL"


def gerar_dados():
    """Cria os registros sintéticos de locais urbanos."""
    registros = []

    for identificador in range(1, QUANTIDADE_LOCAIS + 1):
        tipo_local = np.random.choice(TIPOS_LOCAIS)

        # Características físicas simuladas do local.
        largura = round(np.random.uniform(0.45, 2.50), 2)
        inclinacao = round(np.random.uniform(0, 18), 2)

        # Recursos de acessibilidade disponíveis.
        rampa = int(np.random.choice([0, 1]))
        piso_tatil = int(np.random.choice([0, 1]))
        piso_regular = int(np.random.choice([0, 1]))
        sinalizacao_acessivel = int(np.random.choice([0, 1]))

        # Quantidade de relatos e usuários distintos simulados.
        quantidade_relatos = int(np.random.randint(0, 16))
        usuarios_distintos = int(
            np.random.randint(0, quantidade_relatos + 1)
        )

        # Nota demonstrativa baseada nas características do local.
        nota = 20

        nota += 20 if largura >= 1.20 else 0
        nota += 15 if inclinacao <= 5 else 0
        nota += 10 if rampa == 1 else 0
        nota += 10 if piso_tatil == 1 else 0
        nota += 10 if piso_regular == 1 else 0
        nota += 10 if sinalizacao_acessivel == 1 else 0

        # Pequena variação para evitar uma base totalmente determinística.
        nota += int(np.random.randint(-8, 9))
        nota = max(0, min(100, nota))

        situacao = classificar_acessibilidade(nota)

        # Coordenadas simuladas em três regiões de São Paulo.
        regiao = np.random.choice([0, 1, 2])
        deslocamentos = [
            (0.015, 0.015),
            (-0.025, 0.020),
            (0.020, -0.025),
        ]

        deslocamento_lat, deslocamento_lon = deslocamentos[regiao]

        latitude = CENTRO_LATITUDE + deslocamento_lat + np.random.normal(
            0, 0.004
        )
        longitude = CENTRO_LONGITUDE + deslocamento_lon + np.random.normal(
            0, 0.004
        )

        registros.append({
            "id_local": identificador,
            "tipo_local": tipo_local,
            "largura_calcada_m": largura,
            "inclinacao_graus": inclinacao,
            "rampa": rampa,
            "piso_tatil": piso_tatil,
            "piso_regular": piso_regular,
            "sinalizacao_acessivel": sinalizacao_acessivel,
            "quantidade_relatos": quantidade_relatos,
            "usuarios_distintos": usuarios_distintos,
            "nota_acessibilidade": nota,
            "situacao_acessibilidade": situacao,
            "latitude": round(latitude, 6),
            "longitude": round(longitude, 6),
            "origem_dados": "SINTETICO_SIMULADO",
        })

    return pd.DataFrame(registros)


def main():
    """Gera e salva o arquivo CSV na pasta dados do projeto."""
    # Localiza a raiz do projeto a partir deste próprio arquivo.
    raiz_projeto = Path(__file__).resolve().parent.parent
    pasta_dados = raiz_projeto / "dados"

    pasta_dados.mkdir(parents=True, exist_ok=True)

    caminho_csv = pasta_dados / "locais_acessibilidade.csv"

    df = gerar_dados()
    df.to_csv(caminho_csv, index=False, encoding="utf-8-sig")

    print("Base sintética gerada com sucesso!")
    print(f"Quantidade de registros: {len(df)}")
    print(f"Arquivo salvo em: {caminho_csv}")
    print("\nDistribuição das classificações:")
    print(df["situacao_acessibilidade"].value_counts())
    print("\nOrigem dos dados:")
    print(df["origem_dados"].value_counts())


if __name__ == "__main__":
    main()
