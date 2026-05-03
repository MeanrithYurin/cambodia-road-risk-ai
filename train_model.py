import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    xgboost_available = True
except Exception:
    xgboost_available = False


DATA_PATH = "data/raw/cambodia_road_accident_large_dataset.csv"
MODEL_PATH = "models/best_risk_model.pkl"

os.makedirs("models", exist_ok=True)

df = pd.read_csv(DATA_PATH)

features = [
    "province_en",
    "day_of_week",
    "hour",
    "time_period_en",
    "vehicle_type_en",
    "weather_en",
    "road_type_en",
    "road_condition_en",
    "lighting_en",
    "driver_age_group_en",
    "helmet_used_en",
    "seatbelt_used_en",
    "speeding_en",
    "alcohol_related_en",
    "holiday_en"
]

target = "risk_level_en"

X = df[features]
y = df[target]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

categorical_features = [
    "province_en",
    "day_of_week",
    "time_period_en",
    "vehicle_type_en",
    "weather_en",
    "road_type_en",
    "road_condition_en",
    "lighting_en",
    "driver_age_group_en",
    "helmet_used_en",
    "seatbelt_used_en",
    "speeding_en",
    "alcohol_related_en",
    "holiday_en"
]

numeric_features = ["hour"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
}

if xgboost_available:
    models["XGBoost"] = XGBClassifier(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=6,
        random_state=42,
        eval_metric="mlogloss"
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

best_model = None
best_name = None
best_accuracy = 0

for name, model in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("=" * 60)
    print(name)
    print("Accuracy:", round(acc * 100, 2), "%")
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    ))

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = pipeline
        best_name = name

joblib.dump(
    {
        "model": best_model,
        "label_encoder": label_encoder,
        "features": features,
        "best_model_name": best_name,
        "accuracy": best_accuracy
    },
    MODEL_PATH
)

print("=" * 60)
print("Best model:", best_name)
print("Best accuracy:", round(best_accuracy * 100, 2), "%")
print("Saved model to", MODEL_PATH)