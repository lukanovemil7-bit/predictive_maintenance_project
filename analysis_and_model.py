from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/predictive_maintenance.csv")
UCI_CSV_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
TARGET_CANDIDATES = ["Machine failure", "Target"]
ID_AND_LEAKAGE_COLUMNS = ["UDI", "UID", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
NUMERIC_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_COLUMNS = ["Type"]


def generate_demo_dataset(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Создает автономный демонстрационный набор с той же схемой, что и AI4I 2020."""
    import numpy as np

    rng = np.random.default_rng(seed)
    types = rng.choice(["L", "M", "H"], size=n, p=[0.5, 0.3, 0.2])
    counters = {"L": 0, "M": 0, "H": 0}
    product_ids = []
    for product_type in types:
        counters[product_type] += 1
        start = {"L": 50000, "M": 10000, "H": 70000}[product_type]
        product_ids.append(f"{product_type}{start + counters[product_type]:05d}")

    air_walk = np.cumsum(rng.normal(0, 0.08, size=n))
    air_temperature = 300 + 2 * (air_walk - air_walk.mean()) / air_walk.std()
    process_walk = np.cumsum(rng.normal(0, 0.05, size=n))
    process_offset = 10 + (process_walk - process_walk.mean()) / process_walk.std()
    low_delta_idx = rng.choice(n, size=int(n * 0.025), replace=False)
    process_offset[low_delta_idx] = rng.normal(8.2, 0.25, size=len(low_delta_idx))
    process_temperature = air_temperature + process_offset

    torque = rng.normal(40, 10, size=n)
    torque = np.clip(torque, 3, 76)
    rotational_speed = 1650 - 5.2 * (torque - 40) + rng.normal(0, 90, size=n)
    outlier_idx = rng.choice(n, size=int(n * 0.03), replace=False)
    rotational_speed[outlier_idx] += rng.choice([-350, 420], size=len(outlier_idx))
    rotational_speed = np.clip(rotational_speed, 1000, 2900).round().astype(int)

    wear = []
    current_wear = 0
    for product_type in types:
        increment = {"L": 2, "M": 3, "H": 5}[product_type]
        current_wear += increment
        if current_wear > 253 or rng.random() < 0.015:
            current_wear = rng.integers(0, 20)
        wear.append(current_wear)
    tool_wear = np.array(wear)

    twf = ((tool_wear >= 200) & (tool_wear <= 240) & (rng.random(n) < 0.28)).astype(int)
    hdf = (((process_temperature - air_temperature) < 8.6) & (rotational_speed < 1380)).astype(int)
    power = torque * rotational_speed * 2 * np.pi / 60
    pwf = ((power < 3500) | (power > 9000)).astype(int)
    thresholds = np.array([11000 if t == "L" else 12000 if t == "M" else 13000 for t in types])
    osf = ((tool_wear * torque) > thresholds).astype(int)
    rnf = (rng.random(n) < 0.001).astype(int)
    machine_failure = ((twf + hdf + pwf + osf + rnf) > 0).astype(int)

    return pd.DataFrame(
        {
            "UDI": range(1, n + 1),
            "Product ID": product_ids,
            "Type": types,
            "Air temperature [K]": air_temperature.round(1),
            "Process temperature [K]": process_temperature.round(1),
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque.round(1),
            "Tool wear [min]": tool_wear.astype(int),
            "Machine failure": machine_failure,
            "TWF": twf,
            "HDF": hdf,
            "PWF": pwf,
            "OSF": osf,
            "RNF": rnf,
        }
    )


@st.cache_data(show_spinner=False)
def load_default_dataset() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    try:
        return pd.read_csv(UCI_CSV_URL)
    except Exception:
        return generate_demo_dataset()


def find_target_column(data: pd.DataFrame) -> str:
    for column in TARGET_CANDIDATES:
        if column in data.columns:
            return column
    raise ValueError("В данных не найден целевой столбец: ожидается 'Machine failure' или 'Target'.")


def prepare_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, str]:
    data = data.copy()
    data.columns = [column.strip() for column in data.columns]
    target_column = find_target_column(data)

    missing_required = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing_required:
        raise ValueError(f"В датасете отсутствуют обязательные признаки: {missing_required}")

    columns_to_drop = [column for column in ID_AND_LEAKAGE_COLUMNS if column in data.columns]
    model_data = data.drop(columns=columns_to_drop)
    X = model_data[FEATURE_COLUMNS]
    y = model_data[target_column].astype(int)
    return X, y, model_data, target_column


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
        ]
    )


def get_models() -> Dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, Pipeline], pd.DataFrame, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    trained_models: Dict[str, Pipeline] = {}
    rows = []
    curves = {}

    for model_name, estimator in get_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        trained_models[model_name] = pipeline
        rows.append(
            {
                "Модель": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1-score": f1_score(y_test, y_pred, zero_division=0),
                "ROC-AUC": roc_auc,
            }
        )
        curves[model_name] = {
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": roc_auc,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, zero_division=0),
        }

    metrics = pd.DataFrame(rows).sort_values(by="F1-score", ascending=False).reset_index(drop=True)
    return trained_models, metrics, curves


def plot_class_distribution(y: pd.Series):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=y, ax=ax)
    ax.set_title("Распределение целевой переменной")
    ax.set_xlabel("Machine failure")
    ax.set_ylabel("Количество")
    return fig


def plot_correlation(model_data: pd.DataFrame, target_column: str):
    numeric_columns = [column for column in NUMERIC_COLUMNS + [target_column] if column in model_data.columns]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(model_data[numeric_columns].corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Корреляция числовых признаков")
    return fig


def plot_roc_curves(curves: dict):
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name, curve in curves.items():
        ax.plot(curve["fpr"], curve["tpr"], label=f"{model_name}: AUC={curve['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Случайная модель")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-кривые моделей")
    ax.legend()
    return fig


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title("Confusion Matrix")
    return fig


def show_prediction_form(model: Pipeline):
    st.subheader("Предсказание по новым данным")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            product_type = st.selectbox("Тип продукта", ["L", "M", "H"], index=0)
            air_temp = st.number_input("Air temperature [K]", value=300.0, step=0.1)
        with col2:
            process_temp = st.number_input("Process temperature [K]", value=310.0, step=0.1)
            rotational_speed = st.number_input("Rotational speed [rpm]", value=1500, step=10)
        with col3:
            torque = st.number_input("Torque [Nm]", value=40.0, step=0.1)
            tool_wear = st.number_input("Tool wear [min]", value=120, min_value=0, step=1)
        submitted = st.form_submit_button("Предсказать")

    if submitted:
        input_data = pd.DataFrame(
            [
                {
                    "Type": product_type,
                    "Air temperature [K]": air_temp,
                    "Process temperature [K]": process_temp,
                    "Rotational speed [rpm]": rotational_speed,
                    "Torque [Nm]": torque,
                    "Tool wear [min]": tool_wear,
                }
            ]
        )
        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0, 1])
        label = "отказ оборудования" if prediction == 1 else "отказ не прогнозируется"
        st.metric("Результат", label)
        st.metric("Вероятность отказа", f"{probability:.2%}")


st.title("Бинарная классификация для предиктивного обслуживания оборудования")
st.write(
    "Приложение загружает датасет, выполняет предобработку, обучает несколько моделей "
    "и прогнозирует вероятность отказа оборудования."
)

with st.sidebar:
    st.header("Данные")
    uploaded_file = st.file_uploader("Загрузите CSV-файл", type="csv")
    use_default = st.checkbox("Использовать датасет из проекта", value=True)

try:
    if uploaded_file is not None:
        raw_data = pd.read_csv(uploaded_file)
        source_label = "загруженный CSV-файл"
    elif use_default:
        raw_data = load_default_dataset()
        source_label = "data/predictive_maintenance.csv или fallback-загрузка"
    else:
        st.info("Загрузите CSV-файл или включите использование датасета из проекта.")
        st.stop()

    X, y, model_data, target_column = prepare_data(raw_data)
except Exception as error:
    st.error(f"Ошибка подготовки данных: {error}")
    st.stop()

st.caption(f"Источник данных: {source_label}")

col_a, col_b, col_c = st.columns(3)
col_a.metric("Количество строк", f"{len(raw_data):,}".replace(",", " "))
col_b.metric("Количество признаков модели", len(FEATURE_COLUMNS))
col_c.metric("Доля отказов", f"{y.mean():.2%}")

with st.expander("Первые строки датасета"):
    st.dataframe(raw_data.head(20), use_container_width=True)

with st.expander("Проверка пропущенных значений"):
    missing = raw_data.isna().sum().reset_index()
    missing.columns = ["Столбец", "Количество пропусков"]
    st.dataframe(missing, use_container_width=True)

viz_col1, viz_col2 = st.columns(2)
with viz_col1:
    st.pyplot(plot_class_distribution(y))
with viz_col2:
    st.pyplot(plot_correlation(model_data, target_column))

st.header("Обучение и оценка моделей")
with st.spinner("Обучение моделей..."):
    models, metrics, curves = train_and_evaluate(X, y)

st.subheader("Сравнение моделей")
st.dataframe(metrics.style.format({
    "Accuracy": "{:.3f}",
    "Precision": "{:.3f}",
    "Recall": "{:.3f}",
    "F1-score": "{:.3f}",
    "ROC-AUC": "{:.3f}",
}), use_container_width=True)

best_model_name = str(metrics.iloc[0]["Модель"])
st.success(f"Лучшая модель по F1-score: {best_model_name}")

st.pyplot(plot_roc_curves(curves))

selected_model_name = st.selectbox("Выберите модель для детального просмотра и предсказания", list(models.keys()), index=list(models.keys()).index(best_model_name))
selected_curve = curves[selected_model_name]
st.subheader(f"Детализация модели: {selected_model_name}")
col_left, col_right = st.columns([1, 1])
with col_left:
    st.pyplot(plot_confusion_matrix(selected_curve["confusion_matrix"]))
with col_right:
    st.text("Classification Report")
    st.code(selected_curve["classification_report"], language="text")

show_prediction_form(models[selected_model_name])
