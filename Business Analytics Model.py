#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
try:
    file_path = r'C:/Users/dell/Desktop/Cleaned Dataset.xlsx'  
    df = pd.read_excel(file_path)
except PermissionError as e:
    print(f"Permission Error: {e}")
    print("Please make sure the file is not open in another application and check the file path.")
except FileNotFoundError as e:
    print(f"File Not Found Error: {e}")
    print("Please check if the file path is correct.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Conducting EDA on the file
if 'df' in locals():
    # Display basic information about the dataset
    print("Basic Information:")
    print(df.info())
    print("\nFirst few rows of the dataset:")
    print(df.head())

    # Check for missing values
    print("\nMissing values in each column:")
    missing_values = df.isnull().sum()
    print(missing_values)

    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    string_columns = df.select_dtypes(include=[object]).columns

    # Impute numeric columns with mean
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # Impute string columns with a placeholder
    df[string_columns] = df[string_columns].fillna('Unknown')

    # Using advanced imputation methods like KNN for numerical columns
    imputer = KNNImputer(n_neighbors=3)
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    # Set the actual name of your target column
    target_column = 'Amount'  # Replace 'Amount' with your actual target column name

    # Verify the target column exists
    if target_column in df.columns:
        print("\nClass Imbalance in Amount:")
        print(df[target_column].value_counts())

        # Feature Engineering and Scaling
        # Select numeric columns for scaling
        df_numeric = df.select_dtypes(include=[np.number])
        
        # Split data into features and target
        X = df_numeric.drop(columns=[target_column])  # Features
        y = df_numeric[target_column]  # Target
        
        # Splitting data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalization
        normalizer = Normalizer()
        X_train_norm = normalizer.fit_transform(X_train)
        X_test_norm = normalizer.transform(X_test)

        # Standardization
        scaler = StandardScaler()
        X_train_stzd = scaler.fit_transform(X_train)
        X_test_stzd = scaler.transform(X_test)

        # Visualization
        # Visualizing data distributions with histograms
        print("\nVisualizing data distributions:")
        df_numeric.hist(figsize=(12, 10))
        plt.show()

        # Boxplot visualization for outlier detection
        print("\nBoxplots for numerical columns:")
        for column in df_numeric.columns:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=df_numeric[column])
            plt.title(f'Boxplot of {column}')
            plt.show()

        # Correlation Matrix
        print("\nCorrelation Matrix:")
        correlation_matrix = df_numeric.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix")
        plt.show()

        # Detect and remove outliers using the IQR method
        def detect_outliers(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            return outliers

        # Detect and remove outliers for all numerical columns
        outlier_indices = []
        for column in df_numeric.columns:
            outliers = detect_outliers(df_numeric, column)
            outlier_indices.extend(outliers.index)

        outlier_indices = list(set(outlier_indices))
        print(f"\nFound {len(outlier_indices)} outliers. Removing them...")
        df_cleaned = df.drop(index=outlier_indices)

        # Calculating Customer Lifetime Value (CLV)
        if 'CustomerID' in df_cleaned.columns and target_column in df_cleaned.columns:
            clv_data = df_cleaned.groupby('CustomerID').agg(
                TotalTransactions=(target_column, 'sum'),
                TotalPurchases=('CustomerID', 'count')
            ).reset_index()

            # Calculate average transaction value and purchase frequency
            clv_data['AvgTransactionValue'] = clv_data['TotalTransactions'] / clv_data['TotalPurchases']
            customer_lifespan = 1  # Adjust based on your model

            # Calculate CLV
            clv_data['CustomerLifetimeValue'] = clv_data['AvgTransactionValue'] * clv_data['TotalPurchases'] * customer_lifespan

            # Display the CLV data
            print("\nCustomer Lifetime Value (CLV) for each customer:")
            print(clv_data[['CustomerID', 'CustomerLifetimeValue']])

            # Save the CLV data to a new Excel file
            clv_file_path = "Customer_Lifetime_Value.xlsx"
            clv_data.to_excel(clv_file_path, index=False)
            print(f"\nCustomer Lifetime Value data saved to '{clv_file_path}'.")

        # Save the cleaned dataset
        cleaned_file_path = "Cleaned_Dataset_Final.xlsx"
        df_cleaned.to_excel(cleaned_file_path, index=False)
        print(f"\nCleaned dataset saved to '{cleaned_file_path}'.")

        # Build and train the model
        model = LinearRegression()
        model.fit(X_train_stzd, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_stzd)

        # Evaluate the model's performance
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"\nModel Evaluation Metrics:")
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r2}")

    else:
        print(f"Target column '{target_column}' not found in the dataset.")


# In[ ]:




