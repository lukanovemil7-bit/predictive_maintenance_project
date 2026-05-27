import streamlit as st

try:
    import reveal_slides as rs
except Exception:
    rs = None

st.title("Презентация проекта")

presentation_markdown = """
# Прогнозирование отказов оборудования

---

## Цель проекта

Разработать модель машинного обучения для бинарной классификации:

- `0` — отказ оборудования не прогнозируется;
- `1` — прогнозируется отказ оборудования.

---

## Датасет

Используется датасет AI4I 2020 Predictive Maintenance Dataset.

Основные признаки:

- тип продукта: L, M, H;
- температура воздуха и процесса;
- скорость вращения;
- крутящий момент;
- износ инструмента.

---

## Предобработка данных

- удалены идентификаторы и признаки утечки целевой информации;
- категориальный признак `Type` закодирован через One-Hot Encoding;
- числовые признаки масштабированы через StandardScaler;
- данные разделены на обучающую и тестовую выборки в пропорции 80/20.

---

## Модели

В приложении обучаются и сравниваются три модели:

1. Logistic Regression;
2. Random Forest;
3. Gradient Boosting.

---

## Метрики качества

Для оценки используются:

- Accuracy;
- Precision;
- Recall;
- F1-score;
- ROC-AUC;
- Confusion Matrix.

---

## Streamlit-приложение

Приложение содержит две страницы:

- анализ данных, обучение моделей и предсказание;
- презентация проекта.

Навигация реализована через `st.navigation` и `st.Page`.

---

## Вывод

Проект демонстрирует полный цикл решения задачи бинарной классификации:

- загрузка данных;
- предобработка;
- обучение моделей;
- сравнение качества;
- интерактивный прогноз отказа оборудования.
"""

with st.sidebar:
    st.header("Настройки презентации")
    theme = st.selectbox("Тема", ["black", "white", "league", "beige", "sky", "night", "serif", "simple", "solarized"], index=1)
    height = st.number_input("Высота слайдов", min_value=350, max_value=900, value=600, step=50)
    transition = st.selectbox("Переход", ["slide", "convex", "concave", "zoom", "none"], index=0)
    plugins = st.multiselect("Плагины", ["highlight", "katex", "mathjax2", "mathjax3", "notes", "search", "zoom"], default=[])

if rs is not None:
    rs.slides(
        presentation_markdown,
        height=height,
        theme=theme,
        config={"transition": transition, "plugins": plugins},
        markdown_props={"data-separator-vertical": "^--$"},
    )
else:
    st.warning("Библиотека streamlit-reveal-slides не установлена. Показан fallback в формате Markdown.")
    st.markdown(presentation_markdown)
