import pandas as pd
import matplotlib.pyplot as plt
import click

@click.command()
@click.option('--data', 'data_path', type=click.Path(exists=True), help='Path to processed Parquet file')
def run_eda(data_path):
    df = pd.read_parquet(data_path)
    monthly = df.groupby('Month')['Sales'].mean()

    plt.figure()
    monthly.plot(kind='bar')
    plt.title('Average Sales by Month')
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.tight_layout()
    plt.savefig('eda_monthly_sales.png')

if __name__ == '__main__':
    run_eda()
