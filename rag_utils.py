import os
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from markdown import markdown
from tqdm import tqdm

# TODO: импортировать модельный вызов LLM и embedder
# from embedder import embed_chunk

logger = logging.getLogger(__name__)

def load_yaml_metadata(md_file_path: Path) -> Dict:
    """Загрузка YAML метаданных из отдельного файла
    
    Ищет файл с тем же именем, но расширением .yaml или .yml
    Например, для docs/test.md ищет docs/test.yaml или docs/test.yml
    
    Args:
        md_file_path: Путь к Markdown файлу
        
    Returns:
        Словарь с метаданными или пустой словарь, если файл не найден
    """
    # Пробуем .yaml сначала
    yaml_path = md_file_path.with_suffix('.yaml')
    if yaml_path.exists():
        try:
            with open(yaml_path, encoding='utf-8') as f:
                metadata = yaml.safe_load(f)
                return metadata if metadata else {}
        except Exception as e:
            # Логируем ошибку, но продолжаем работу
            logger.warning(f"Ошибка при чтении {yaml_path}: {e}")
            return {}
    
    # Если .yaml не найден, пробуем .yml
    yaml_path = md_file_path.with_suffix('.yml')
    if yaml_path.exists():
        try:
            with open(yaml_path, encoding='utf-8') as f:
                metadata = yaml.safe_load(f)
                return metadata if metadata else {}
        except Exception as e:
            logger.warning(f"Ошибка при чтении {yaml_path}: {e}")
            return {}
    
    # Если ни один файл не найден, возвращаем пустой словарь
    return {}

def generate_yaml_metadata(md_file_path: Path, doc_id: str, tags: List[str], overwrite: bool = False) -> Path:
    """Автоматическая генерация YAML файла с метаданными
    
    Создаёт YAML файл с doc_id и тегами. Если файл уже существует и overwrite=False,
    файл не перезаписывается.
    
    Args:
        md_file_path: Путь к Markdown файлу
        doc_id: Идентификатор документа
        tags: Список тегов
        overwrite: Перезаписать существующий YAML файл (по умолчанию False)
        
    Returns:
        Путь к созданному YAML файлу
    """
    yaml_path = md_file_path.with_suffix('.yaml')
    
    # Если файл существует и не нужно перезаписывать, возвращаем путь
    if yaml_path.exists() and not overwrite:
        logger.debug(f"YAML файл уже существует: {yaml_path}, пропускаю генерацию")
        return yaml_path
    
    # Формируем метаданные
    metadata = {
        'doc_id': doc_id,
        'tags': tags if tags else []
    }
    
    # Записываем YAML файл
    try:
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        logger.info(f"✅ Создан YAML файл: {yaml_path}")
        return yaml_path
    except Exception as e:
        logger.error(f"Ошибка при создании YAML файла {yaml_path}: {e}")
        raise

def split_to_chunks(content: str) -> List[str]:
    """Разбить markdown по ## (главы)"""
    lines = content.splitlines()
    chunks = []
    buf = []
    for line in lines:
        if line.startswith('## '):
            if buf:
                chunks.append('\n'.join(buf).strip())
                buf = []
        buf.append(line)
    if buf:
        chunks.append('\n'.join(buf).strip())
    return [ch for ch in chunks if ch]

def parse_all_headers(content: str) -> List[str]:
    """Получение всех заголовков второго уровня (## ...) как список тегов"""
    return [line[3:].strip() for line in content.splitlines() if line.startswith('## ')]

# Заглушка для тегирования: возвращает первый тег для каждого чанка
def tag_chunks(chunks: List[str], tags: List[str]) -> List[str]:
    results = []
    for ch in chunks:
        # TODO: заменить на вызов LLM-matcher
        results.append(tags[0] if tags else 'Misc')
    return results

def process_markdown_docs(docs_dir: Path, auto_generate_yaml: bool = True, overwrite_yaml: bool = False):
    """Обработка всех Markdown документов в директории
    
    Args:
        docs_dir: Директория с Markdown файлами
        auto_generate_yaml: Автоматически генерировать YAML файлы если их нет (по умолчанию True)
        overwrite_yaml: Перезаписывать существующие YAML файлы (по умолчанию False)
        
    Returns:
        Tuple[List[dict], dict]: Список чанков и отчёт об обработке
    """
    all_chunks = []
    report = {"processed": [], "errors": []}

    md_files = list(docs_dir.rglob('*.md'))

    for f in tqdm(md_files, desc='docs'):
        try:
            # Читаем Markdown файл
            with open(f, encoding='utf-8') as fh:
                md_content = fh.read()
            
            # Загружаем YAML метаданные из отдельного файла
            metadata = load_yaml_metadata(f)
            
            # Извлекаем теги из метаданных или парсим заголовки
            tags_from_metadata = metadata.get('tags', [])
            tags_from_headers = parse_all_headers(md_content)
            
            # Если нужно перезаписать YAML, используем имя файла и теги из заголовков
            # Иначе используем данные из YAML, если они есть
            if overwrite_yaml:
                doc_id = f.stem
                tags = tags_from_headers
            else:
                doc_id = metadata.get('doc_id', f.stem)
                if tags_from_metadata:
                    tags = tags_from_metadata if isinstance(tags_from_metadata, list) else [tags_from_metadata]
                else:
                    tags = tags_from_headers
            
            # Автоматически генерируем YAML файл если его нет или нужно обновить
            if auto_generate_yaml:
                # Обновляем doc_id и tags если они изменились или файла нет
                yaml_path = f.with_suffix('.yaml')
                should_generate = (
                    not yaml_path.exists() or  # Файла нет
                    overwrite_yaml or  # Нужно перезаписать
                    metadata.get('doc_id') != doc_id or  # doc_id изменился
                    (not tags_from_metadata and tags_from_headers)  # Теги нужно обновить из заголовков
                )
                
                if should_generate:
                    generate_yaml_metadata(f, doc_id, tags, overwrite=overwrite_yaml or not yaml_path.exists())
            
            # Разбиваем на чанки
            chunks = split_to_chunks(md_content)
            chunk_tags = tag_chunks(chunks, tags)

            for c, t in zip(chunks, chunk_tags):
                chunk_obj = {
                    "text": c,
                    "metadata": {
                        "doc_id": doc_id,
                        "tag": t,
                        "tags": tags
                    },
                    "path": str(f)
                }
                all_chunks.append(chunk_obj)

            report["processed"].append({
                "file": str(f),
                "doc_id": doc_id,
                "tags": tags,
                "chunks": len(chunks)
            })
        except Exception as e:
            report["errors"].append({"file": str(f), "err": str(e)})
    return all_chunks, report
