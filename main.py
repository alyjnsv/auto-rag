import os
import sys
import logging
from pathlib import Path
import typer
from dotenv import load_dotenv
from rag_utils import process_markdown_docs
from vector_store import get_vector_store

app = typer.Typer()

# Логгирование
logging.basicConfig(
    filename='logs/auto_rag.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

@app.command()
def run(
    docs_path: str = typer.Option('docs', help='Путь к папке с документацией'),
    dry_run: bool = typer.Option(False, help='Не загружать в LEANN, только выводить результат'),
    report_file: str = typer.Option('report.json', help='Имя json-отчёта об обработке'),
    index_name: str = typer.Option(None, help='Имя индекса LEANN (по умолчанию используется doc_id первого документа)'),
    no_auto_yaml: bool = typer.Option(False, help='Отключить автоматическую генерацию YAML файлов'),
    overwrite_yaml: bool = typer.Option(False, help='Перезаписывать существующие YAML файлы')
):
    """Основной запуск системы авто-RAG с LEANN"""
    load_dotenv()
    docs_dir = Path(docs_path)
    if not docs_dir.exists():
        logger.error(f'Папка {docs_dir} не найдена!')
        typer.echo(f'Папка {docs_dir} не найдена!')
        sys.exit(1)
    os.makedirs('logs', exist_ok=True)
    logger.info(f'Обработка папки {docs_dir}')

    try:
        auto_generate_yaml = not no_auto_yaml
        chunks, meta_report = process_markdown_docs(
            docs_dir, 
            auto_generate_yaml=auto_generate_yaml,
            overwrite_yaml=overwrite_yaml
        )
        typer.echo(f'Обработано чанков: {len(chunks)}')
        if auto_generate_yaml:
            typer.echo('✅ YAML файлы автоматически сгенерированы/обновлены')
    except Exception as e:
        logger.exception('Ошибка при обработке docs:')
        typer.echo(f'Ошибка: {e}')
        sys.exit(1)

    if not dry_run:
        try:
            # Определяем имя индекса
            if not index_name:
                # Для LEANN используем имя первого документа или общее
                if meta_report.get('processed'):
                    index_name = meta_report['processed'][0].get('doc_id', 'auto-rag-index')
                else:
                    index_name = 'auto-rag-index'
            
            # Создаём LEANN векторную БД
            vector_store = get_vector_store()
            vector_store.upload_chunks(chunks, index_name)
            typer.echo(f'✅ Чанки загружены в LEANN (индекс: {index_name})')
            typer.echo(f'📁 Индекс сохранён в: .leann/indexes/{index_name}')
        except ImportError as e:
            logger.warning(f'LEANN не установлен: {e}')
            typer.echo('⚠️  LEANN не установлен. Пропускаю загрузку.')
            typer.echo('💡 Установите LEANN:')
            typer.echo('   - uv pip install leann (рекомендуется)')
            typer.echo('   - или см. INSTALL_LEANN.md для других способов')
        except Exception as e:
            logger.exception('Ошибка при загрузке в LEANN:')
            typer.echo(f'Ошибка загрузки в LEANN: {e}')
    else:
        typer.echo('[dry_run] Выгрузка в LEANN пропущена')

    import json
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(meta_report, f, ensure_ascii=False, indent=2)
    typer.echo(f'Отчёт сохранён: {report_file}')

if __name__ == '__main__':
    app()
