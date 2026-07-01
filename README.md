# Экзамен: Структурированный вывод (Pydantic)

## Описание задания

В рамках курса **«Generating Structured Outputs»** вы должны написать небольшое приложение,
которое из произвольного текста извлекает валидированные объекты Pydantic без ручного разбора строк.
Текст может описывать либо человека, либо встречу.  
Приложение должно:

1. Определить тип входного текста (человек / встреча).
2. Выбрать подходящую схему и выполнить извлечение с помощью LangChain.
3. Показать результат в виде `model_dump()` и краткой сводки.

---

## Стек

| Библиотека | Версия |
|------------|--------|
| Python     | 3.10+  |
| langchain-core | ≥1.0.0 |
| langchain-openai (или langchain-ollama) | – |
| pydantic   | – |
| python-dotenv | – |

```bash
pip install langchain-core langchain-openai pydantic python-dotenv
```

> **Важно**: используйте `langchain >= 1.0`.  
> Для OpenAI понадобится переменная окружения `OPENAI_API_KEY` (или `OLLAMA_URL`, если вы пользуетесь Ollama).

---

## Архитектура проекта

```
project/
├── main.py            # точка входа, CLI и логика выбора схемы
├── models.py          # Pydantic‑модели PersonInfo и MeetingNotes
├── prompt_templates.py  # PromptTemplate для каждой задачи
└── README.md          # это файл
```

### Файлы

| Файл | Что содержит |
|------|--------------|
| `models.py` | Определения двух Pydantic‑моделей с полями и описаниями (`Field(description=…)`). |
| `prompt_templates.py` | Три PromptTemplate: `PERSON_PROMPT`, `MEETING_PROMPT`, `TYPE_DETECTOR_PROMPT`. |
| `main.py` | Загрузка переменных окружения, создание LLM, парсеров, цепочек и CLI. |

---

## 1. Pydantic‑модели

```python
# models.py
from pydantic import BaseModel, Field
from typing import List, Optional

class PersonInfo(BaseModel):
    name: str = Field(..., description="Имя человека")
    age: Optional[int] = Field(None, description="Возраст (необязательно)")
    profession: str = Field(..., description="Профессия")
    skills: List[str] = Field(..., description="Список навыков")

class MeetingNotes(BaseModel):
    date: str = Field(..., description="Дата встречи в формате YYYY-MM-DD")
    participants: List[str] = Field(..., description="Участники встречи")
    topics: List[str] = Field(..., description="Темы обсуждения")
    decisions: List[str] = Field(..., description="Принятые решения")
    next_steps: List[str] = Field(..., description="Следующие шаги (список строк)")
```

---

## 2. Prompt‑шаблоны

```python
# prompt_templates.py
from langchain_core.prompts import PromptTemplate

PERSON_PROMPT = PromptTemplate(
    input_variables=["text", "format_instructions"],
    template=(
        "Найди в тексте информацию о человеке и выведи JSON, "
        "соответствующий схеме PersonInfo.\n\n"
        "{format_instructions}\n\nТекст: {text}"
    ),
)

MEETING_PROMPT = PromptTemplate(
    input_variables=["text", "format_instructions"],
    template=(
        "Найди в тексте информацию о встрече и выведи JSON, "
        "соответствующий схеме MeetingNotes.\n\n"
        "{format_instructions}\n\nТекст: {text}"
    ),
)

TYPE_DETECTOR_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=(
        "Определи, описывается ли в тексте человек или встреча. "
        "Ответь только одним словом: 'person' или 'meeting'.\n\nТекст: {text}"
    ),
)
```

---

## 3. Основная логика (`main.py`)

```python
# main.py
import os
import sys
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

from models import PersonInfo, MeetingNotes
from prompt_templates import PERSON_PROMPT, MEETING_PROMPT, TYPE_DETECTOR_PROMPT

# 1️⃣ Загрузка переменных окружения
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ Не найден API‑ключ OpenAI. Установите переменную OPENAI_API_KEY.")
    sys.exit(1)

# 2️⃣ Инициализация LLM
llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0)

# 3️⃣ Создание парсеров
person_parser = PydanticOutputParser(pydantic_object=PersonInfo)
meeting_parser = PydanticOutputParser(pydantic_object=MeetingNotes)

# 4️⃣ Конструируем цепочки
person_chain = PERSON_PROMPT | llm | person_parser
meeting_chain = MEETING_PROMPT | llm | meeting_parser
type_detector_chain = TYPE_DETECTOR_PROMPT | llm

def detect_type(text: str) -> str:
    """Определяем тип текста."""
    result = type_detector_chain.invoke({"text": text})
    return result.strip().lower()

def extract_person(text: str):
    return person_chain.invoke({
        "text": text,
        "format_instructions": person_parser.get_format_instructions()
    })

def extract_meeting(text: str):
    return meeting_chain.invoke({
        "text": text,
        "format_instructions": meeting_parser.get_format_instructions()
    })

# 5️⃣ CLI
EXAMPLES = {
    "person": (
        "Анна, 28 лет, Python-разработчик. Навыки: FastAPI, Docker."
    ),
    "meeting": (
        "Встреча 2023-10-12 с участниками Иван и Мария. "
        "Обсуждались темы: CI/CD, масштабирование. "
        "Принято решение внедрить GitHub Actions. "
        "Следующие шаги: подготовить документацию, назначить ответственных."
    ),
}

def main():
    if len(sys.argv) > 1 and sys.argv[1] in EXAMPLES:
        text = EXAMPLES[sys.argv[1]]
        print(f"🔹 Используем пример «{sys.argv[1]}»")
    else:
        print("📥 Введите текст (Ctrl+D для завершения):")
        text = sys.stdin.read().strip()
        if not text:
            print("❌ Пустой ввод.")
            return

    obj_type = detect_type(text)
    print(f"🕵️‍♂️ Определён тип: {obj_type}")

    try:
        if obj_type == "person":
            result = extract_person(text)
            print("\n✅ PersonInfo:")
            print(result.model_dump(indent=2))
        elif obj_type == "meeting":
            result = extract_meeting(text)
            print("\n✅ MeetingNotes:")
            print(result.model_dump(indent=2))
        else:
            print("❌ Не удалось определить тип. Попробуйте другой текст.")
    except Exception as e:
        print(f"⚠️ Ошибка при парсинге: {e}")

if __name__ == "__main__":
    main()
```

### Как запустить

```bash
# 1️⃣ Установите зависимости
pip install -r requirements.txt   # (или просто pip install langchain-core langchain-openai pydantic python-dotenv)

# 2️⃣ Создайте файл .env с ключом OpenAI
echo "OPENAI_API_KEY=sk-..." > .env

# 3️⃣ Запустите приложение
python main.py person      # пример о человеке
python main.py meeting     # пример о встрече
python main.py             # ввод из stdin
```

---

## 4. Проверка результата

После успешного запуска вы увидите что-то вроде:

```text
🔹 Используем пример «person»
🕵️‍♂️ Определён тип: person

✅ PersonInfo:
{
  "name": "Анна",
  "age": 28,
  "profession": "Python-разработчик",
  "skills": [
    "FastAPI",
    "Docker"
  ]
}
```

или

```text
🔹 Используем пример «meeting»
🕵️‍♂️ Определён тип: meeting

✅ MeetingNotes:
{
  "date": "2023-10-12",
  "participants": [
    "Иван",
    "Мария"
  ],
  "topics": [
    "CI/CD",
    "масштабирование"
  ],
  "decisions": [
    "внедрить GitHub Actions"
  ],
  "next_steps": [
    "подготовить документацию",
    "назначить ответственных"
  ]
}
```

---

## Критерии зачёта

| Компонент | Проверяется |
|-----------|-------------|
| **Модели** | `PersonInfo` и `MeetingNotes` с полями и описаниями (`Field(description=…)`). |
| **Извлечение** | Используется `PydanticOutputParser` (или `with_structured_output`) без ручного разбора. |
| **Маршрутизация** | Перед извлечением определяется тип текста через отдельный промпт. |
| **CLI** | Возможность выбора примера или ввода из stdin, вывод `model_dump()` и краткая сводка. |
| **Сдача** | LangChain ≥1.0, README, валидированный объект на выходе. |

---

## Полезные ссылки

- [LangChain Structured Output](https://docs.langchain.com/oss/python/langchain/structured-output)
- [Pydantic Docs](https://docs.pydantic.dev/)
- [OpenAI API Key Setup](https://platform.openai.com/account/api-keys)

--- 

Удачной сдачи! 🚀
