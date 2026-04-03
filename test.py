import requests

def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=OOCIBMBBTXN9C2CZ"
    r = requests.get(url)

    return r.json()

print(get_stock_price("AAPL"))