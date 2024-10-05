import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load the dataset from the specified file path.
    """
    return pd.read_csv("../data/data.csv")

def overview(df):
    """
    Generator function that yields the structure of the dataset,
    including the number of rows, columns, and data types.
    """
    # Yield the DataFrame info
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    yield info_str

def summary_statistics(df):
    """
    Display summary statistics for the dataset.
    """
    print(df.describe())

def distribution_numerical_features(df):
    """
    Visualize the distribution of numerical features.
    """
    df.select_dtypes(include=['float64', 'int64']).hist(bins=30, figsize=(10, 8))
    plt.tight_layout()
    plt.show()

def distribution_categorical_features(df):

    """
    Visualize the distribution of categorical features using bar plots.
    """
    group1 = ["TransactionStartTime", "CustomerId", "SubscriptionId", "AccountId", "BatchId", "TransactionId"]

# Get categorical columns from the DataFrame
    categorical_columns = df.select_dtypes(include=['object']).columns

# Iterate through each categorical column
    for column in categorical_columns:
        plt.figure(figsize=(10, 4))
    
    # Check if the column is in group1
        if column in group1:
            sns.countplot(data=df.sample(frac=0.1), x=column)
            plt.title(f'Distribution of {column}')
            plt.xticks(ticks=plt.gca().get_xticks()[::250], rotation=45)
        else:
            sns.countplot(data=df.sample(frac=0.1), x=column)
            plt.title(f'Distribution of {column}')
            plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def correlation_analysis(df):
    """
    Display a heatmap showing the correlation matrix for numerical features.
    """
    numerical_column=df.select_dtypes(include=["float","int"])
    corr_matrix = numerical_column.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

def identify_missing_values(df):
    """
    Identify missing values and visualize them using a heatmap.
    """
    print(df.isnull().sum())
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()

def handle_missing_data(df):
    """
    Handles missing data in the DataFrame by filling:
    - Categorical columns with the mode (most frequent value)
    - Numerical columns with the mean
    """
     # Handle categorical columns (filling missing values with mode)
    categorical_columns = df.select_dtypes(include="object")
    for col in categorical_columns.columns:
        categorical_columns[col] = categorical_columns[col].fillna(categorical_columns[col].mode()[0])

    # Handle numerical columns (filling missing values with mean)
    numerical_columns = df.select_dtypes(include=["float", "int"])
    for col in numerical_columns.columns:
        numerical_columns[col] = numerical_columns[col].fillna(numerical_columns[col].mean())

    # Merge categorical and numerical columns back into the original dataframe
    df[categorical_columns.columns] = categorical_columns
    df[numerical_columns.columns] = numerical_columns

    return df

def detect_outliers(df):
    """
    Detect outliers using box plots for numerical features.
    """
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, x=column)
        plt.title(f'Boxplot of {column}')
        plt.show()