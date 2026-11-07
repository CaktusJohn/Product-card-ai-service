import json
import logging
import os

from dotenv import load_dotenv
from fastapi import Body, FastAPI
from fastapi import HTTPException
from openai import APIError, OpenAI, RateLimitError
from pydantic import BaseModel, Field


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
            "Манометр из ТЗ": {
                "summary": "Пример товара из технического задания",
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
            }
        }
    ),
) -> ProductDraft:
    missing_data = []
    if not product.specs:
        missing_data.append("Характеристики товара")
    if not product.description.strip():
        missing_data.append("Описание или фрагмент технического паспорта")

    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Не задан LLM_API_KEY в .env")

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", "https://api.mistral.ai/v1"),
    )
    model = os.getenv("LLM_MODEL")

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
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=messages,
        )
        generated = json.loads(response.choices[0].message.content or "")

        title_length = len(generated["seo_title"])
        description_length = len(generated["seo_description"])
        if title_length > 70 or not 800 <= description_length <= 1200:
            logger.warning(
                "Первая генерация не прошла проверку длины: title=%d, description=%d; запрашиваю исправление",
                title_length,
                description_length,
            )
            corrections = []
            if title_length > 70:
                corrections.append(f"seo_title: получено {title_length}, максимум 70 символов")
            if description_length < 800 or description_length > 1200:
                corrections.append(
                    f"seo_description: получено {description_length}, требуется 800–1200 символов"
                )
            messages.extend(
                [
                    {"role": "assistant", "content": json.dumps(generated, ensure_ascii=False)},
                    {
                        "role": "user",
                        "content": (
                            "Исправь только поля с нарушением длины: "
                            + "; ".join(corrections)
                            + ". Сохрани подтверждённость фактов и смысл. "
                            "Для описания стремись к 950–1100 символам, считая пробелы. "
                            "Верни полный JSON с теми же ключами."
                        ),
                    },
                ]
            )
            response = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=messages,
            )
            generated = json.loads(response.choices[0].message.content or "")
            attempt = "после повторной генерации"
        else:
            attempt = "с первой попытки"

        title_length = len(generated["seo_title"])
        description_length = len(generated["seo_description"])
        if title_length > 70 or not 800 <= description_length <= 1200:
            logger.error(
                "Генерация не прошла проверку длины %s: title=%d, description=%d",
                attempt,
                title_length,
                description_length,
            )
            raise HTTPException(
                status_code=502,
                detail="Модель не соблюла ограничения длины SEO-полей после повторной генерации.",
            )
        logger.info(
            "Генерация успешна %s: title=%d, description=%d",
            attempt,
            title_length,
            description_length,
        )

        return ProductDraft(
            seo_title=generated["seo_title"],
            seo_description=generated["seo_description"],
            search_queries=generated["search_queries"],
            commercial_block=generated["commercial_block"],
            characteristics=product.specs or {"Данные": "данных нет"},
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
