import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df):
    print("Dataset Info:")
    print(df.info())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nClass Distribution:")
    print(df['Class'].value_counts(normalize=True) * 100)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Amount'], bins=50)
    plt.title('Transaction Amount Distribution')
    plt.xlabel('Amount')
    plt.ylabel('Frequency')
    plt.show()

    corr = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

if __name__ == "__main__":
    run_eda()