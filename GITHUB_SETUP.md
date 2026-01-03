# Инструкция по публикации на GitHub

## Шаг 1: Создайте репозиторий на GitHub

1. Перейдите на https://github.com/new
2. Заполните форму:
   - **Repository name**: `auto-rag`
   - **Description**: `Автоматическая система RAG-ингестинга Markdown документации с LEANN`
   - **Visibility**: Public или Private (на ваш выбор)
   - **НЕ** создавайте README, .gitignore или license (они уже есть в проекте)
3. Нажмите "Create repository"

## Шаг 2: Подключите remote и запушьте код

После создания репозитория выполните следующие команды:

```bash
cd /Users/a1234/auto-rag

# Добавьте remote (замените YOUR_USERNAME на ваш GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/auto-rag.git

# Переименуйте ветку в main (если нужно)
git branch -M main

# Запушьте код
git push -u origin main
```

## Альтернативный способ через SSH

Если у вас настроен SSH ключ:

```bash
git remote add origin git@github.com:YOUR_USERNAME/auto-rag.git
git branch -M main
git push -u origin main
```

## Проверка

После успешного пуша проверьте:
- https://github.com/YOUR_USERNAME/auto-rag
- README должен отображаться корректно
- Все файлы должны быть на месте
