import pandas as pd
import click

def engineer_features(df):
    # Example: Extract day of week, month, and year from date
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    # Convert categorical variables
    df = pd.get_dummies(df, columns=['Product'], drop_first=True)

    return df

@click.command()
@click.option('--input', type=click.Path(exists=True), help='Path to raw sales data CSV')
@click.option('--output', type=click.Path(), help='Path to save processed features')
def main(input, output):
    df = pd.read_csv(input)
    df_features = engineer_features(df)
    df_features.to_parquet(output, index=False)

if __name__ == '__main__':
    main()
