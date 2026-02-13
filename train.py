import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("Loading dataset...")

# -------------------------------
# 1️⃣ Load Dataset
# -------------------------------
df = pd.read_csv("data/Dataset.csv")

# -------------------------------
# 2️⃣ Cleaning
# -------------------------------

df.drop_duplicates(inplace=True)

df["Delivery_person_Ratings"] = pd.to_numeric(
    df["Delivery_person_Ratings"], errors="coerce"
)
df["Delivery_person_Age"] = pd.to_numeric(
    df["Delivery_person_Age"], errors="coerce"
)

df = df[(df["Restaurant_latitude"] != 0) &
        (df["Restaurant_longitude"] != 0)]

df = df[(df["Delivery_person_Age"] >= 18) &
        (df["Delivery_person_Age"] <= 60)]

df = df[(df["Delivery_person_Ratings"] >= 1) &
        (df["Delivery_person_Ratings"] <= 5)]

df = df[df["Delivery Time_taken(min)"] > 0]

# -------------------------------
# 3️⃣ Feature Engineering
# -------------------------------

# Extract City
df["City"] = df["Delivery_person_ID"].str.split("RES").str[0]

# Haversine Distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians,
                                 [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

df["distance_km"] = haversine(
    df["Restaurant_latitude"],
    df["Restaurant_longitude"],
    df["Delivery_location_latitude"],
    df["Delivery_location_longitude"]
)

# Efficiency Score
df["efficiency_score"] = (
    df["Delivery_person_Age"] *
    df["Delivery_person_Ratings"]
)

# Interaction Features (Boost performance)
df["distance_squared"] = df["distance_km"] ** 2
df["rating_distance_interaction"] = (
    df["Delivery_person_Ratings"] * df["distance_km"]
)
df["age_distance_interaction"] = (
    df["Delivery_person_Age"] * df["distance_km"]
)

# -------------------------------
# 4️⃣ Target Encoding for City
# -------------------------------

city_mean = df.groupby("City")["Delivery Time_taken(min)"].mean()
df["City_encoded"] = df["City"].map(city_mean)

# -------------------------------
# 5️⃣ Drop Unnecessary Columns
# -------------------------------

df.drop(columns=[
    "ID",
    "Delivery_person_ID",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "City"
], inplace=True)

# -------------------------------
# 6️⃣ Prepare Data
# -------------------------------

X = df.drop("Delivery Time_taken(min)", axis=1)

# Log transform target
y = np.log1p(df["Delivery Time_taken(min)"])

# One-hot encode remaining categorical
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 7️⃣ Tuned XGBoost Model
# -------------------------------

print("Training model...")

model = xgb.XGBRegressor(
    n_estimators=1500,
    learning_rate=0.01,
    max_depth=10,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# -------------------------------
# 8️⃣ Evaluate Model
# -------------------------------

log_pred = model.predict(X_test)
pred = np.expm1(log_pred)
y_actual = np.expm1(y_test)

rmse = np.sqrt(mean_squared_error(y_actual, pred))
r2 = r2_score(y_actual, pred)

print("\nOptimized Model Performance:")
print("RMSE:", rmse)
print("R2 Score:", r2)

# -------------------------------
# 9️⃣ Save Model + Columns
# -------------------------------

joblib.dump(model, "models/model.pkl")
joblib.dump(X.columns, "models/columns.pkl")

print("\nModel and feature columns saved successfully!")
