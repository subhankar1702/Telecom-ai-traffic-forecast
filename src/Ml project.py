# Telecom AI Project
# Traffic Forecasting + Congestion Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# --------------------------------------------------
# Load Data
# --------------------------------------------------

def load_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Date','Cell_ID'])
    return df


# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

def feature_engineering(df):

    # Lag Features
    df['lag_1'] = df.groupby('Cell_ID')['DL_Volume_GB'].shift(1)
    df['lag_24'] = df.groupby('Cell_ID')['DL_Volume_GB'].shift(24)
    df['lag_168'] = df.groupby('Cell_ID')['DL_Volume_GB'].shift(168)

    # Rolling Features
    df['rolling_mean_24'] = df.groupby('Cell_ID')['DL_Volume_GB'].transform(
        lambda x: x.rolling(24).mean()
    )

    df['rolling_mean_168'] = df.groupby('Cell_ID')['DL_Volume_GB'].transform(
        lambda x: x.rolling(168).mean()
    )

    # Time Features
    df['hour'] = df['Date'].dt.hour
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month

    # Telecom KPI Feature
    df['Traffic_per_User'] = df['DL_Volume_GB'] / (df['Active_Users'] + 1)

    df = df.dropna()

    return df


# --------------------------------------------------
# Traffic Forecasting Model
# --------------------------------------------------

def train_traffic_models(X_train,y_train,X_test,y_test):

    models = {

        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ),

        "XGBoost": xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

    }

    results = []

    for name,model in models.items():

        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test,y_pred)
        rmse = np.sqrt(mean_squared_error(y_test,y_pred))

        results.append({
            "Model":name,
            "MAE":mae,
            "RMSE":rmse
        })

        print(f"{name} -> MAE:{mae:.3f} RMSE:{rmse:.3f}")

    results_df = pd.DataFrame(results)

    return models["XGBoost"],results_df


# --------------------------------------------------
# Congestion Prediction Model
# --------------------------------------------------

def train_congestion_model(X_train,y_train,X_test,y_test):

    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8
    )

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:",accuracy_score(y_test,y_pred))
    print("ROC AUC:",roc_auc_score(y_test,y_pred))
    print(classification_report(y_test,y_pred))

    return model


# --------------------------------------------------
# Feature Importance
# --------------------------------------------------

def plot_feature_importance(model,features):

    importance = model.feature_importances_

    feature_imp = pd.DataFrame({
        "Feature":features,
        "Importance":importance
    })

    feature_imp = feature_imp.sort_values(
        by="Importance",
        ascending=False
    )

    print(feature_imp)

    plt.figure(figsize=(10,6))
    plt.barh(feature_imp['Feature'],feature_imp['Importance'])
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()
    plt.show()


# --------------------------------------------------
# Main Pipeline
# --------------------------------------------------

def main():

    data_path = "data/telecom_traffic_1yr_200cells.csv.gz"

    df = load_data(data_path)

    print("Data Loaded:",df.shape)

    df = feature_engineering(df)

    print("Feature Engineering Done:",df.shape)


    # --------------------------
    # Traffic Forecasting
    # --------------------------

    target = 'DL_Volume_GB'

    features = [
        'lag_1','lag_24','lag_168',
        'rolling_mean_24','rolling_mean_168',
        'PRB_Util','Active_Users',
        'Weekend','Event_Flag',
        'hour','day_of_week','month',
        'Traffic_per_User'
    ]

    X = df[features]
    y = df[target]

    split = int(len(df)*0.8)

    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    traffic_model,results = train_traffic_models(
        X_train,y_train,X_test,y_test
    )

    print("\nModel Comparison")
    print(results)

    plot_feature_importance(traffic_model,features)


    # --------------------------
    # Congestion Prediction
    # --------------------------

    df['Congestion'] = (df['PRB_Util'] > 80).astype(int)

    y = df['Congestion']

    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    congestion_model = train_congestion_model(
        X_train,y_train,X_test,y_test
    )

    plot_feature_importance(congestion_model,features)


    # --------------------------
    # Save Models
    # --------------------------

    joblib.dump(traffic_model,"models/traffic_forecasting_model.pkl")
    joblib.dump(congestion_model,"models/congestion_prediction_model.pkl")

    print("\nModels Saved Successfully")


# --------------------------------------------------

if __name__ == "__main__":
    main()