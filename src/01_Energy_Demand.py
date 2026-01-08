"""Energy demand prediction (MW) using ML models.

It:
- Loads the raw CSV (day, hour, temperature, mw)
- Builds features: weekday (0=Mon..6=Sun) and minute-of-day (hour*60)
- Splits into train/test
- Trains Linear Regression, Decision Tree Regressor, Random Forest Regressor
- Evaluates with cross-validation (RMSE) on train and final RMSE/R2 on test
- Generates and saves the main figures

Usage:
    01_Energy_Demand.py --csv data/demanda-maxima-de-mendoza-2022.csv --outdir results
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score



# -----------------------------
# Data loading & preprocessing
# -----------------------------
def data_loading(root):
    df = pd.read_csv(
    root,
    encoding = "latin1",
    delimiter = ';',
    decimal = ',')
    return df

def preprocessing(df):
    columns = ['day', 'hour', 'temperature', 'mw']
    df.columns = columns

    df['day'] = pd.to_datetime(df['day'], dayfirst=True)
    df['weekday'] = df['day'].dt.weekday
    df['hour'] = df['hour'].str.slice(0, -3).astype(int)
    df['minute'] = df['hour'] * 60
    df = df[['weekday', 'minute', 'temperature', 'mw']]
    df.to_csv("results/series.txt", index=0)
    return df


def save_fig(fig_id, tight_layout=True, fig_extension="png",resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension,dpi=resolution)

def plot_histograms(df):
    plt.rc('font', size = 14)
    plt.rc('axes', labelsize = 14, titlesize = 14)
    plt.rc('legend', fontsize = 14)
    plt.rc('legend', fontsize = 14)
    plt.rc('xtick', labelsize = 10)
    plt.rc('ytick', labelsize = 10)
    df.hist(bins = 50, figsize = (12,8))
    save_fig("histograms")

def target_selection(df, target_name = 'mw'):
    #Seleccionamos la columna con el objetivo:
    target_name = target_name
    y = df[target_name]
    #Los atributos:
    X = df.drop([target_name], axis = 1)
    return X, y
        
# Feature engineering
def plot_demand_per_day(df):
    mean_by_weekday = (
        df
        .groupby('weekday')['mw']
        .mean()
    )
    plt.figure(figsize=(8, 6))
    plt.bar(mean_by_weekday.index, mean_by_weekday.values)
    plt.title('mean demand per day')
    plt.xlabel('weekday')
    plt.ylabel('mean demand(MW)')
    plt.grid(axis='y')
    plt.tight_layout()
    save_fig('demand_per_day')

def plot_demand_vs_temperature(df):
    plt.figure(figsize=(8, 6))

    # Scatter
    plt.scatter(
        df['temperature'],
        df['mw'],
        alpha=0.6,
        label='observations'
    )

    # Tendencia media
    mean_by_temp = (
        df
        .assign(temp_round=df['temperature'].round())
        .groupby('temp_round')['mw']
        .mean()
    )

    plt.plot(
        mean_by_temp.index,
        mean_by_temp.values,
        color='red',
        marker='o',
        linewidth=2,
        label='mean demand'
    )

    plt.title('demand vs. temperature')
    plt.xlabel('temperature (ºC)')
    plt.ylabel('demand (MW)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_fig("demand_vs_temperature")


# -----------------------------
# Modeling
# -----------------------------

def model_training(X_train, y_train, random_state, dt_max_depth=5, dt_min_samples_leaf=5, rf_n_estimators=300, rf_max_depth=8, rf_min_samples_leaf=5, cv = 5):
    lin_model = LinearRegression()
    dt_model = DecisionTreeRegressor(max_depth=dt_max_depth, min_samples_leaf=dt_min_samples_leaf, random_state=random_state)
    rf_model = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth, min_samples_leaf=rf_min_samples_leaf, random_state=random_state)
    lin_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)
    rf_model.fit(X_train,y_train)
    lin_scores = cross_val_score(lin_model, X_train, y_train, scoring = 'neg_mean_squared_error', cv=cv)
    lin_rmse_scores = np.sqrt(-lin_scores)
    dt_scores = cross_val_score(dt_model, X_train, y_train, scoring = 'neg_mean_squared_error', cv=cv)
    dt_rmse_scores = np.sqrt(-dt_scores)
    rf_scores = cross_val_score(rf_model, X_train, y_train, scoring = 'neg_mean_squared_error', cv = cv)
    rf_rmse_scores = np.sqrt(-rf_scores)
    return dt_model, lin_rmse_scores, dt_rmse_scores, rf_rmse_scores

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())



# -----------------------------
# Main
# -----------------------------
def main() -> None:
    global IMAGES_PATH
    IMAGES_PATH = Path() / "images" / "classification"
    IMAGES_PATH.mkdir(parents = True, exist_ok = True)
    DATA_PATH = r"data/demanda-maxima-de-mendoza-2022.csv" 
    RANDOM_STATE = 42
    
    df = data_loading(DATA_PATH)
    data = preprocessing(df)
    plot_histograms(data)
    X,y = target_selection(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= RANDOM_STATE)
    train_data = X_train.copy()
    train_data['mw'] = y_train

    #Feature Engineering
    plot_demand_per_day(train_data)
    plot_demand_vs_temperature(train_data)

    #Training
    dt_model, lin_metrics, dt_metrics, rf_metrics = model_training(X_train, y_train, RANDOM_STATE)
    display_scores(lin_metrics)
    display_scores(dt_metrics)
    display_scores(rf_metrics)

    #Evaluation in test
    y_test_pred = dt_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    print("Model Evaluation")
    print(f"Testing RMSE: {test_rmse:.2f}")
    print(f"Testing R2: {test_r2:.2f}")

    #Visualization of results
    plt.figure(figsize = (8,6))
    plt.scatter(y_test, y_test_pred, alpha = 0.5, label = "Predecido vs. Actual")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(),y_test.max()], 'r--', label = 'Ideal_fit')
    plt.xlabel('real demand')
    plt.ylabel('predicted demand')
    plt.title("predicted demand (mw) vs real demand")
    plt.legend()
    plt.grid(True)
    save_fig('dt_results_demand_prediction')


    #Visualization of decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(
        dt_model,
        feature_names=X_train.columns,  # si X_train es DataFrame
        filled=True,
        rounded=True,
        fontsize=10
    )
    save_fig('decision_tree_regressor')
    plt.title("Decision Tree Regressor")


if __name__ == "__main__":
    main()