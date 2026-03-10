import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("EXCHANGE_API_KEY")

if not api_key:
    raise ValueError("API key not found. Check your .env file")


def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    url = f'https://v6.exchangerate-api.com/v6/{api_key}/pair/{base_currency}/{target_currency}'

    response = requests.get(url)

    print("Status Code:", response.status_code)
    print("Response:", response.text)

    data = response.json()

    if response.status_code != 200:
        raise Exception(f"API Error: {data}")

    return data["conversion_rate"]


print(get_conversion_factor("USD", "INR"))