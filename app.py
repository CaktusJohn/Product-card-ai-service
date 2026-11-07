import json
import logging
import os
import time

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException
from openai import APIError, OpenAI, RateLimitError
from pydantic import BaseModel, Field, field_validator


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("product_card")


app = FastAPI(title="Product Card AI Service")

SYSTEM_PROMPT = """
Ты готовишь черновики карточек для B2B-магазина измерительных приборов.
Пиши по-русски, профессионально и понятно для технического специалиста и закупщика.
Используй спокойный деловой стиль, без чрезмерной рекламы и повторов.
Не используй оценочные заявления вроде «инновационный», «новое поколение»,
«бескомпромиссный», «повышенная точность», «улучшенная надёжность» или
«принципиально новый уровень», если источник не приводит проверяемые факты,
которые подтверждают такое сравнение. Не представляй рекламные оценки как факты.

Факты можно брать только из названия, производителя и описания во входных данных.
Считай вход закрытым списком разрешённых фактов: нельзя выводить новые сведения
из того, что обычно верно для товаров такого типа. Например, не приписывай
манометру трубопроводы, гидравлические или пневматические системы, измеряемые
среды и условия эксплуатации, если это прямо не указано. Не приписывай модулю
отрасли, безопасность, рост эффективности, точность или интеграцию с системами,
если это прямо не указано.
Не добавляй технические параметры, числа, функции, сертификаты, совместимость,
преимущества или области применения, если они прямо не подтверждены источником.
Не делай выводов о точности, надёжности или выгоде только по типу товара.
Характеристики отдельным списком не формируй.

Сформируй SEO-описание в такой последовательности, используя только подтверждённое:
1. Кратко представь товар, его тип и назначение.
2. Опиши указанные во входе функции и особенности.
3. Объясни практическую пользу характеристик, только если она прямо следует из данных.
4. Укажи применение только тогда, когда оно указано во входе.
5. Заверши кратким коммерческим выводом для подходящего покупателя.
Если данных мало, сообщи об этом прямо, не заполняй объём выдумками.
Не добавляй условные оговорки в скобках и не маскируй предположение словами
«может применяться», «подходит для» или «актуально для». Такие формулировки тоже
считаются утверждениями о товаре и допустимы только при прямом подтверждении.

Верни только JSON с ключами seo_title, seo_description, search_queries,
commercial_block.
- seo_title: до 70 символов, включая пробелы; цель — 50–65 символов.
- seo_description: 800–1200 символов, включая пробелы; цель — 950–1100.
- search_queries: 3–5 естественных запросов по названию, типу и подтверждённому назначению.
- commercial_block: кому и для какой задачи подходит товар, если это подтверждено;
  иначе укажи, что данных для такого вывода недостаточно.
Перед ответом проверь длину заголовка и описания.
""".strip()


class ProductInput(BaseModel):
    name: str = Field(min_length=1)
    manufacturer: str = Field(min_length=1)
    specs: dict[str, str] = Field(default_factory=dict)
    description: str = ""

    @field_validator("name", "manufacturer")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Поле не должно быть пустым")
        return value


class ProductDraft(BaseModel):
    seo_title: str = Field(max_length=70)
    seo_description: str = Field(min_length=800, max_length=1200)
    characteristics: dict[str, str]
    search_queries: list[str] = Field(min_length=3, max_length=5)
    commercial_block: str
    missing_data: list[str]


@app.post("/generate", response_model=ProductDraft)
def generate_product_card(
    product: ProductInput = Body(
        openapi_examples={
            "Манометр DM 02": {
                "summary": "Пример товара",
                "value": {
                    "name": "Манометр показывающий DM 02",
                    "manufacturer": "Манотомь",
                    "specs": {
                        "диапазон измерений": "0–1,6 МПа",
                        "класс точности": "1,5",
                        "присоединение": "G1/2",
                        "диаметр корпуса": "100 мм",
                    },
                    "description": "Прибор предназначен для измерения избыточного давления...",
                },
            },
            "ОВЕН МВ110-224.8А": {
                "summary": "Пример товара",
                "value": {
                    "name": "ОВЕН МВ110-224.8А",
                    "manufacturer": "ОВЕН",
                    "specs": {
                        "тип": "модуль ввода аналоговых сигналов",
                        "количество входных каналов": "8",
                        "тип интерфейса": "RS-485",
                        "напряжение питания": "24 В DC / 220 В AC",
                    },
                    "description": (
                        "Модуль аналогового ввода ОВЕН МВ110-224.8А (H/W v2.0) "
                        "предназначен для мониторинга параметров в системах автоматизации. "
                        "Работает с сигналами термопар, "
                        "тензодатчиков и преобразователей давления."
                    ),
                },
            },
        }
    ),
) -> ProductDraft:
    missing_data = []
    if not product.specs:
        missing_data.append("Характеристики товара не переданы")
        characteristics = {"Данные": "данных нет"}
    else:
        characteristics = {}
        for name, value in product.specs.items():
            if value.strip():
                characteristics[name] = value
            else:
                characteristics[name] = "данных нет"
                missing_data.append(f"Характеристика «{name}» не заполнена")
    if not product.description.strip():
        missing_data.append("Описание или фрагмент технического паспорта")

    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Не задан LLM_API_KEY в .env")

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", "https://api.mistral.ai/v1"),
    )
    model = os.getenv("LLM_MODEL", "mistral-small-latest")

    try:
        logger.info("Генерация начата: товар=%r, модель=%s", product.name, model)
        user_prompt = json.dumps(
            {
                "name": product.name,
                "manufacturer": product.manufacturer,
                "description": product.description,
            },
            ensure_ascii=False,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        for attempt in range(3):
            response = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=messages,
            )
            generated = json.loads(response.choices[0].message.content or "")
            title_length = len(generated["seo_title"])
            description_length = len(generated["seo_description"])

            if title_length <= 70 and 800 <= description_length <= 1200:
                logger.info(
                    "Генерация успешна: попытка=%d, title=%d, description=%d",
                    attempt + 1,
                    title_length,
                    description_length,
                )
                break

            if attempt == 2:
                logger.error(
                    "Генерация не прошла проверку длины после 3 попыток: title=%d, description=%d",
                    title_length,
                    description_length,
                )
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Модель не соблюла ограничения длины после 3 попыток. "
                        f"SEO-заголовок: {title_length} символов, "
                        f"SEO-описание: {description_length} символов."
                    ),
                )

            corrections = []
            if title_length > 70:
                corrections.append(f"seo_title: сейчас {title_length}, максимум 70 символов")
            if not 800 <= description_length <= 1200:
                corrections.append(
                    f"seo_description: сейчас {description_length}, нужно 800–1200 символов"
                )
            logger.warning(
                "Ответ не прошёл проверку длины: title=%d, description=%d; запрашиваю исправление",
                title_length,
                description_length,
            )
            delay = 2**attempt
            logger.info("Пауза перед повторным запросом: %d сек.", delay)
            time.sleep(delay)
            messages.extend(
                [
                    {"role": "assistant", "content": json.dumps(generated, ensure_ascii=False)},
                    {
                        "role": "user",
                        "content": (
                            "Исправь только поля с нарушением длины: "
                            + "; ".join(corrections)
                            + ". Сохрани подтверждённость фактов и смысл. "
                            "Для SEO-описания стремись к 950–1100 символам с пробелами. "
                            "Верни полный JSON с теми же ключами."
                        ),
                    },
                ]
            )

        return ProductDraft(
            seo_title=generated["seo_title"],
            seo_description=generated["seo_description"],
            search_queries=generated["search_queries"],
            commercial_block=generated["commercial_block"],
            characteristics=characteristics,
            missing_data=missing_data,
        )
    except RateLimitError as error:
        logger.warning("LLM-провайдер вернул ограничение запроса: %s", error)
        raise HTTPException(status_code=429, detail=str(error)) from error
    except (APIError, ValueError, TypeError, IndexError, KeyError) as error:
        logger.exception("Ошибка генерации или разбора ответа модели")
        raise HTTPException(
            status_code=502,
            detail=f"Не удалось получить корректный ответ модели: {error}",
        ) from error
