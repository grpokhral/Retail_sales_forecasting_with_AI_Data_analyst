import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import click

@click.command()
@click.option('--features', 'features_path', type=click.Path(exists=True))
@click.option('--output', 'model_path', type=click.Path())
def train_model(features_path, model_path):
    df = pd.read_parquet(features_path)
    X = df.drop(columns=['Sales', 'Date'])
    y = df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"Test RMSE: {rmse:.2f}")

    joblib.dump(model, model_path)

if __name__ == '__main__':
    train_model()
