import os
from typing import List
import logging

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logging.warning("openai не установлен. Embedder будет использовать заглушки.")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning("python-dotenv не установлен. Переменные окружения не загружены.")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Настройка API
if HAS_OPENAI:
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
    elif OPENROUTER_API_KEY:
        openai.api_key = OPENROUTER_API_KEY
    else:
        logging.warning("Нет ключа для OpenAI/OpenRouter. Embedder будет возвращать заглушки.")


# Можно реализовать через модель text-embedding-3-large или аналог

def embed_chunk(text: str, model: str = "text-embedding-3-large") -> List[float]:
    """Вычисление embedding для одного чанка"""
    if HAS_OPENAI and openai.api_key:
        try:
            response = openai.Embedding.create(
                input=[text],
                model=model
            )
            return response['data'][0]['embedding']
        except Exception as e:
            logging.warning(f"Ошибка embedding: {e}. Возвращаю пустой вектор.")
            return [0.0] * 1024
    else:
        # Заглушка, если нет ключа или пакета
        return [float(hash(text) % 1000) / 1000.0] * 10
