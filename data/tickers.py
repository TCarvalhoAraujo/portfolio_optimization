"""
tickers.py
==========
Listas pré-definidas de ações do mercado americano organizadas por setor.
Facilita a composição de portfólios diversificados para as simulações.

Uso:
    from data.tickers import SP500_LARGE_CAP, get_diversified_portfolio
"""

# ─────────────────────────────────────────────────────────────────────────────
# LISTAS POR SETOR (S&P 500 — empresas com alta liquidez e dados confiáveis)
# ─────────────────────────────────────────────────────────────────────────────

TECHNOLOGY = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "NVDA",  # NVIDIA
    "GOOGL", # Alphabet
    "META",  # Meta Platforms
    "AVGO",  # Broadcom
    "ORCL",  # Oracle
    "CRM",   # Salesforce
    "AMD",   # Advanced Micro Devices
    "INTC",  # Intel
    "QCOM",  # Qualcomm
    "TXN",   # Texas Instruments
    "AMAT",  # Applied Materials
    "MU",    # Micron Technology
    "LRCX",  # Lam Research
    "ADBE",  # Adobe
]

CONSUMER_DISCRETIONARY = [
    "AMZN",  # Amazon
    "TSLA",  # Tesla
    "HD",    # Home Depot
    "MCD",   # McDonald's
    "NKE",   # Nike
    "SBUX",  # Starbucks
    "TGT",   # Target
    "LOW",   # Lowe's
    "BKNG",  # Booking Holdings
    "GM",    # General Motors
]

HEALTHCARE = [
    "JNJ",   # Johnson & Johnson
    "UNH",   # UnitedHealth Group
    "LLY",   # Eli Lilly
    "PFE",   # Pfizer
    "ABBV",  # AbbVie
    "MRK",   # Merck
    "TMO",   # Thermo Fisher
    "DHR",   # Danaher
    "BMY",   # Bristol-Myers Squibb
    "AMGN",  # Amgen
]

FINANCIALS = [
    "JPM",   # JPMorgan Chase
    "BAC",   # Bank of America
    "V",     # Visa
    "MA",    # Mastercard
    "WFC",   # Wells Fargo
    "GS",    # Goldman Sachs
    "MS",    # Morgan Stanley
    "BLK",   # BlackRock
    "C",     # Citigroup
    "AXP",   # American Express
]

ENERGY = [
    "XOM",   # ExxonMobil
    "CVX",   # Chevron
    "COP",   # ConocoPhillips
    "SLB",   # SLB (Schlumberger)
    "EOG",   # EOG Resources
    "MPC",   # Marathon Petroleum
    "TPL",   # Texas Pacific Land Corporation
]

CONSUMER_STAPLES = [
    "PG",    # Procter & Gamble
    "KO",    # Coca-Cola
    "PEP",   # PepsiCo
    "COST",  # Costco
    "WMT",   # Walmart
    "PM",    # Philip Morris
    "MO",    # Altria
    "CL",    # Colgate-Palmolive
]

INDUSTRIALS = [
    "CAT",   # Caterpillar
    "DE",    # Deere & Company
    "HON",   # Honeywell
    "UPS",   # UPS
    "GE",    # GE Aerospace
    "MMM",   # 3M
]

COMMUNICATION = [
    "NFLX",  # Netflix
    "DIS",   # Disney
    "T",     # AT&T
    "VZ",    # Verizon
    "CMCSA", # Comcast
    "TMUS",  # T-Mobile
]

UTILITIES = [
    "NEE",   # NextEra Energy
    "DUK",   # Duke Energy
    "SO",    # Southern Company
    "D",     # Dominion Energy
    "AEP",   # American Electric Power
]

REAL_ESTATE = [
    "AMT",   # American Tower
    "PLD",   # Prologis
    "EQIX",  # Equinix
    "CCI",   # Crown Castle
    "SPG",   # Simon Property Group
]

DEFENSE = [
    "LMT",   # Lockheed Martin
    "NOC",   # Northrop Grumman
    "GD",    # General Dynamics
    "BA",    # Boeing
    "RTX",   # RTX Corporation
]

# ─────────────────────────────────────────────────────────────────────────────
# PORTFÓLIOS PRÉ-DEFINIDOS
# ─────────────────────────────────────────────────────────────────────────────

# Top 20 por capitalização de mercado
SP500_TOP20 = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "AVGO", "JPM",   "LLY",
    "V",    "UNH",  "JNJ",  "XOM",   "MA",
    "HD",   "PG",   "COST", "NFLX",  "ORCL",
]

# 30 ações: diversificação por setor
SP500_DIVERSIFIED_30 = [
    # Tech (8)
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMD", "ORCL", "CRM",
    # Financials (5)
    "JPM", "V", "MA", "BAC", "GS",
    # Healthcare (5)
    "JNJ", "UNH", "LLY", "PFE", "MRK",
    # Consumer (4)
    "AMZN", "TSLA", "HD", "MCD",
    # Energy (3)
    "XOM", "CVX", "COP",
    # Staples (3)
    "PG", "KO", "WMT",
    # Industrials (2)
    "CAT", "HON",
]

# 50 ações: portfólio amplo e equilibrado (proposta principal do projeto)
# Cobre 11 setores incluindo defesa, energia expandida e finanças diversificadas
MY_DIVERSIFIED_50 = [
    # Technology (8) — hardware, software, semicondutores, nuvem
    "AAPL", "MSFT", "NVDA", "GOOGL", "AVGO", "AMD", "ORCL", "ADBE",
    # Financials (8) — bancos, pagamentos, corretoras, gestoras
    "JPM", "V", "MA", "BAC", "GS", "MS", "BLK", "AXP",
    # Healthcare (5) — seguro, farmacêutica, equipamentos médicos
    "UNH", "LLY", "JNJ", "PFE", "TMO",
    # Consumer Discretionary (4) — e-commerce, varejo, restaurantes, vestuário
    "AMZN", "HD", "MCD", "NKE",
    # Consumer Staples (3) — defensivos, baixa volatilidade
    "PG", "KO", "WMT",
    # Energy (6) — petróleo integrado, E&P, serviços, refino
    "XOM", "CVX", "TPL", "SLB", "MPC",
    # Defense & Aerospace (5) — contratos governamentais, baixa correlação com tech
    "RTX", "LMT", "NOC", "GD", "BA",
    # Industrials (4) — maquinário, logística, conglomerado
    "HON", "CAT", "DE", "UPS",
    # Communication (3) — streaming, telecoms, entretenimento
    "NFLX", "TMUS", "DIS",
    # Utilities (2) — defensivo, exposição a energia limpa
    "NEE", "AEP",
    # Real Estate (2) — REITs de infraestrutura e logística
    "AMT", "PLD",
]

# 50 ações: portfólio amplo
SP500_50 = SP500_DIVERSIFIED_30 + [
    "AVGO", "TXN", "QCOM",           # Tech extra
    "WFC",  "MS",  "BLK",            # Financials extra
    "TMO",  "DHR", "ABBV",           # Healthcare extra
    "NKE",  "SBUX", "COST",          # Consumer extra
    "NEE",  "DUK",                   # Utilities
    "AMT",  "PLD",                   # Real Estate
    "NFLX", "DIS",                   # Communication
    "DE",   "UPS",                   # Industrials extra
]


def get_diversified_portfolio(n: int = 30) -> list[str]:
    """
    Retorna um portfólio diversificado com n ações.

    Parâmetros
    ----------
    n : número de ações desejadas (entre 10 e 100)

    Retorna
    -------
    Lista de tickers balanceada entre setores
    """
    if n <= 10:
        return SP500_TOP20[:n]
    elif n <= 20:
        return SP500_TOP20[:n]
    elif n <= 30:
        return SP500_DIVERSIFIED_30[:n]
    elif n <= 50:
        return SP500_50[:n]
    else:
        # Para n > 50, combina todos os setores
        all_tickers = (
            TECHNOLOGY[:20] + FINANCIALS[:10] + HEALTHCARE[:10] +
            CONSUMER_DISCRETIONARY[:10] + ENERGY[:6] +
            CONSUMER_STAPLES[:8] + INDUSTRIALS[:8] +
            COMMUNICATION[:6] + UTILITIES[:5] + REAL_ESTATE[:5]
        )
        return all_tickers[:n]


# ─────────────────────────────────────────────────────────────────────────────
# TODOS OS SETORES COMO DICIONÁRIO
# ─────────────────────────────────────────────────────────────────────────────

SECTORS = {
    "Technology":             TECHNOLOGY,
    "Consumer Discretionary": CONSUMER_DISCRETIONARY,
    "Healthcare":             HEALTHCARE,
    "Financials":             FINANCIALS,
    "Energy":                 ENERGY,
    "Consumer Staples":       CONSUMER_STAPLES,
    "Industrials":            INDUSTRIALS,
    "Communication":          COMMUNICATION,
    "Utilities":              UTILITIES,
    "Real Estate":            REAL_ESTATE,
    "Defense":               DEFENSE,
}


if __name__ == "__main__":
    print("Portfólio diversificado (30 ações):")
    print(get_diversified_portfolio(30))
    print()
    print("Setores disponíveis:")
    for sector, tickers in SECTORS.items():
        print(f"  {sector:<25}: {len(tickers)} ações")