"""
Интеграция с LEANN - локальной векторной БД с экономией памяти до 97%.
"""
import os
import logging
from typing import List, Dict
from pathlib import Path

# LEANN
try:
    from leann import Builder, Searcher
    HAS_LEANN = True
except ImportError:
    HAS_LEANN = False


class LEANNStore:
    """Реализация для LEANN - локальная векторная БД с экономией памяти"""
    
    def __init__(self, index_dir: str = ".leann/indexes"):
        if not HAS_LEANN:
            error_msg = (
                "LEANN не установлен или недоступен для вашей системы.\n"
                "Попробуйте:\n"
                "  1. uv pip install leann (рекомендуется)\n"
                "  2. Установка из исходников: git clone https://github.com/yichuan-w/LEANN.git\n"
                "Подробнее см. INSTALL_LEANN.md"
            )
            raise ImportError(error_msg)
        
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.builder = None
        self.searcher = None
        self.current_index_name = None
    
    def upload_chunks(self, chunks: List[Dict], index_name: str) -> None:
        """Загружает чанки в LEANN индекс"""
        logging.info(f"Начинаю загрузку {len(chunks)} чанков в LEANN (индекс: {index_name})...")
        
        index_path = self.index_dir / index_name
        
        # Если индекс уже существует, удаляем его для пересоздания
        if index_path.exists():
            import shutil
            logging.info(f"Удаляю существующий индекс: {index_path}")
            shutil.rmtree(index_path)
        
        try:
            # Создаём Builder для нового индекса
            # LEANN использует graph-based структуру с selective recomputation
            embedding_model = os.getenv('LEANN_EMBEDDING_MODEL', 'facebook/contriever')
            backend = os.getenv('LEANN_BACKEND', 'hnsw')  # 'hnsw' или 'diskann'
            
            self.builder = Builder(
                index_path=str(index_path),
                embedding_model=embedding_model,
                backend=backend
            )
            
            # Добавляем чанки с метаданными
            for i, c in enumerate(chunks):
                text = c['text']
                metadata = c['metadata']
                
                # LEANN поддерживает метаданные для фильтрации
                # Метод add_text принимает текст и опциональные метаданные
                try:
                    self.builder.add_text(
                        text=text,
                        metadata=metadata
                    )
                except Exception as e:
                    logging.warning(f"Ошибка при добавлении чанка {i}: {e}")
                    # Пробуем без метаданных
                    self.builder.add_text(text=text)
                
                if (i + 1) % 100 == 0:
                    logging.info(f"Обработано {i + 1} из {len(chunks)} чанков")
            
            # Финальная сборка индекса
            # LEANN строит графовую структуру вместо хранения всех эмбеддингов
            logging.info("Сборка LEANN индекса (это может занять время)...")
            self.builder.build()
            logging.info(f"✅ LEANN индекс создан: {index_path}")
            logging.info(f"💾 Экономия хранилища: ~97% по сравнению с традиционными векторными БД")
            
            self.current_index_name = index_name
            
        except Exception as e:
            logging.error(f"Ошибка при создании LEANN индекса: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Поиск в LEANN индексе
        
        LEANN использует selective recomputation - вычисляет эмбеддинги
        только для узлов в пути поиска, что обеспечивает быстрый поиск
        при минимальном использовании памяти.
        """
        if self.current_index_name is None:
            raise ValueError("Индекс не создан. Сначала загрузите чанки.")
        
        index_path = self.index_dir / self.current_index_name
        
        if not index_path.exists():
            raise ValueError(f"Индекс не найден: {index_path}")
        
        try:
            if self.searcher is None:
                self.searcher = Searcher(index_path=str(index_path))
            
            # LEANN автоматически вычисляет эмбеддинги для запроса
            # и использует graph traversal для поиска
            results = self.searcher.search(query=query, top_k=top_k)
            return results
        except Exception as e:
            logging.error(f"Ошибка при поиске в LEANN: {e}")
            raise


def get_vector_store(index_dir: str = ".leann/indexes") -> LEANNStore:
    """
    Создаёт экземпляр LEANN векторной БД
    
    Args:
        index_dir: Директория для хранения индексов
    
    Returns:
        Экземпляр LEANNStore
    """
    return LEANNStore(index_dir=index_dir)
