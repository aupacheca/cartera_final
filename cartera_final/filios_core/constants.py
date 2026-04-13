"""Constantes de columnas CSV/tablas y parámetros globales de formato."""
from __future__ import annotations

DECIMALS_POSITION = 8
MIN_POSITION = 10 ** -DECIMALS_POSITION

CSV_ENCODING = "latin-1"
CSV_DECIMAL = ","
CSV_SEP = ","

DIVIDENDOS_COLUMNS = [
    "type", "date", "time", "ticker", "ticker_Yahoo", "isin", "nombre", "positionType", "positionCountry",
    "positionCurrency", "positionExchange", "broker", "positionNumber", "currency", "quantity",
    "quantityCurrency", "comission", "comissionCurrency", "exchangeRate", "comissionBaseCurrency",
    "autoFx", "total", "totalBaseCurrency", "originRetention", "neto", "netoBaseCurrency",
    "destinationRetentionBaseCurrency", "totalNeto", "totalNetoBaseCurrency", "retentionReturned",
    "retentionReturnedBaseCurrency", "unrealizedDestinationRetentionBaseCurrency",
    "netoWithReturnBaseCurrency", "originRetentionLossBaseCurrency", "description",
]

MOVIMIENTOS_COLUMNS = [
    "date", "time", "ticker", "ticker_Yahoo", "isin", "name", "positionType", "positionCountry",
    "positionCurrency", "positionExchange", "broker", "type", "positionNumber", "price",
    "comission", "comissionCurrency", "destinationRetentionBaseCurrency", "taxes", "taxesCurrency",
    "exchangeRate", "positionQuantity", "autoFx", "switchBuyPosition", "switchBuyPositionType",
    "switchBuyPositionNumber", "switchBuyExchangeRate", "switchBuyBroker", "spinOffBuyPosition",
    "spinOffBuyPositionNumber", "spinOffBuyPositionAllocation", "brokerTransferNewBroker",
    "total", "totalBaseCurrency", "totalWithComission", "totalWithComissionBaseCurrency",
]

MOVIMIENTOS_CRIPTOS_COLUMNS = [
    "date", "time", "ticker", "ticker_Yahoo", "isin", "name", "positionType", "positionCountry",
    "positionCurrency", "positionExchange", "broker", "type", "positionNumber", "price",
    "comission", "comissionCurrency", "destinationRetentionBaseCurrency", "taxes", "taxesCurrency",
    "exchangeRate", "positionQuantity", "autoFx", "switchBuyPosition", "switchBuyPositionType",
    "switchBuyPositionNumber", "switchBuyExchangeRate", "switchBuyBroker", "spinOffBuyPosition",
    "spinOffBuyPositionNumber", "spinOffBuyPositionAllocation", "brokerTransferNewBroker",
    "total", "totalBaseCurrency", "totalWithComission", "totalWithComissionBaseCurrency",
    "positionCustomType", "description",
]

CRYPTO_BROKER_IDS = {
    "67b242abada74321db44e91b": "Binance",
    "67c8ac4deb09ee2b1a4121d3": "Tangem",
}

CRYPTO_TICKER_NAMES = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "XRP": "Ripple", "SOL": "Solana",
    "BNB": "Binance", "TRX": "Tron", "AVAX": "Avalanche", "HBAR": "Hedera",
    "ADA": "Cardano", "DOT": "Polkadot", "LINK": "Chainlink", "MATIC": "Polygon",
    "DOGE": "Dogecoin", "UNI": "Uniswap", "ATOM": "Cosmos", "LTC": "Litecoin",
}
