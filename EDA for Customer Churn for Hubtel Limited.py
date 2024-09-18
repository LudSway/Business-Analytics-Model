#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Attempt to load the Excel file
try:
    file_path = r'C:/Users/dell/Desktop/Cleaned Dataset.xlsx'  
    data = pd.read_excel(file_path)
except PermissionError as e:
    print(f"Permission Error: {e}")
    print("Please make sure the file is not open in another application and check the file path.")
except FileNotFoundError as e:
    print(f"File Not Found Error: {e}")
    print("Please check if the file path is correct.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Conducting EDA on the file
if 'data' in locals():
    # Step 1: Display basic information about the dataset
    print("Basic Information:")
    print(data.info())
    print("\nFirst few rows of the dataset:")
    print(data.head())

    # Step 2: Check for missing values
    print("\nMissing values in each column:")
    missing_values = data.isnull().sum()
    print(missing_values)

    # Step 3: Generate statistical summary
    print("\nStatistical Summary:")
    stat_summary = data.describe()
    print(stat_summary)

    # Step 4: Visualize data distributions with histograms
    print("\nVisualizing data distributions:")
    data.hist(figsize=(12, 10))
    plt.show()

    # Step 5: Boxplot visualization for outlier detection
    print("\nBoxplots for numerical columns:")
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=data[column])
        plt.title(f'Boxplot of {column}')
        plt.show()

    # Step 6: Correlation Matrix
    print("\nCorrelation Matrix:")
    numerical_data = data.select_dtypes(include=['float64', 'int64'])
    if not numerical_data.empty:
        correlation_matrix = numerical_data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix")
        plt.show()
    else:
        print("No numerical columns found for correlation matrix.")

    # Step 7: Detect and remove outliers using the IQR method
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
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        outliers = detect_outliers(data, column)
        outlier_indices.extend(outliers.index)

    outlier_indices = list(set(outlier_indices))
    print(f"\nFound {len(outlier_indices)} outliers. Removing them...")
    data_cleaned = data.drop(outlier_indices)

    # Step 8: Calculate Customer Lifetime Value (CLV)
    # Using columns: 'CustomerID', 'Amount', and 'TransactionDate'

    # Calculate total Amount Spent per customer
    clv_data = data_cleaned.groupby('CustomerID').agg(
        TotalTransactions=('Amount', 'sum'),
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
    clv_data.to_excel("Customer_Lifetime_Value.xlsx", index=False)
    print(f"\nCustomer Lifetime Value data saved to 'Customer_Lifetime_Value.xlsx'.")

    # Step 9: Save the cleaned dataset
    cleaned_file_path = "Cleaned_Dataset_Final.xlsx"
    data_cleaned.to_excel(cleaned_file_path, index=False)
    print(f"\nCleaned dataset saved to '{cleaned_file_path}'.")

