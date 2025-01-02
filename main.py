from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import xgboost as XGBRegressor
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from typing import List
import json


app = FastAPI()



class InputData(BaseModel):
    data: list  
    num_days: int




# PREPROCESSING
def clean_column_names(df):
    def clean_name(name):
        name = name.lower()
        name = re.sub(r'[^\w]', '', name)
        name = name.replace(' ', '')
        name = name.replace('_', '')
        return name
    
    df.columns = [clean_name(col) for col in df.columns]
    df = df.loc[:, ~df.columns.str.contains('cust')]

    return df

def standardize_date_columns(df):    
    date_columns = [col for col in df.columns if 'date' in col]
    
    if not date_columns:
        raise ValueError("Dataset tidak memiliki kolom yang mengandung kata 'date'.")

    standardized_dates = []

    for col in date_columns:
        df[col] = df[col].astype(str) 
        if (df[col].str.contains('/').any() and df[col].str.contains('-').any()):
            df[col] = df[col].str.replace('-', '/')
        try:
            df[col] = pd.to_datetime(df[col]) 
        except Exception:
            try:
                df[col] = pd.to_datetime(df[col], format="%d/%m/%Y")
            except Exception:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception:
                    continue
        
        standardized_dates.append(df[col])
        
        # endfor

    df.drop(columns=date_columns, inplace=True)
    
    df['TEMP DATE'] = pd.concat(standardized_dates, axis=1).bfill(axis=1).iloc[:, 0]
    
    df = df.sort_values('TEMP DATE').reset_index(drop=True)
    return df

def standardize_product_columns(df):
    prioritize_name = ['sku', 'description', 'name', 'detail', 'product', 'article', 'menu']
    found_columns = []
    
    for name in prioritize_name:
        for col in df.columns:
            if name in col:
                found_columns.append(col)
        
        if len(found_columns) != 0:
            break
    
    delete_col = []
    
    if len(found_columns) > 1:
        for i in range(len(found_columns)):
            if 'category' in found_columns[i]:
                delete_col.append(i)
                
    for i in range(len(delete_col)):
        found_columns.pop(delete_col[i])
    
    
    df = df.rename(columns={f'{found_columns[0]}': 'TEMP PRODUCT'})
    
    found_columns = []
    return df

def standardize_quantity_columns(df):
    prioritize_name = ['quantity', 'qty', 'units']
    found_columns = []
    
    for name in prioritize_name:
        for col in df.columns:
            if name in col:
                found_columns.append(col)
        
        if len(found_columns) != 0:
            break
    
    if len(found_columns) == 0:
        df['TEMP QUANTITY'] = 1
        df['TEMP QUANTITY'] = df['TEMP QUANTITY'].astype(float)

        return df
    
    df = df.rename(columns={f'{found_columns[0]}': 'TEMP QUANTITY'})
    df['TEMP QUANTITY'] = df['TEMP QUANTITY'].astype(float)
    
    df = df[df['TEMP QUANTITY'] > 0]

    
    found_columns = []
    return df

def standardize_price_columns(df):
    prioritize_name = ['money', 'price']
    found_columns = []
    
    for name in prioritize_name:
        for col in df.columns:
            if name in col:
                found_columns.append(col)
        
        if len(found_columns) != 0:
            break
    
    delete_col = []
    
    if len(found_columns) > 1:
        for i in range(len(found_columns)):
            if 'total' in found_columns[i]:
                delete_col.append(i)
    
                    
    for i in range(len(delete_col)):
            found_columns.pop(delete_col[i])
                
    df = df.rename(columns={f'{found_columns[0]}': 'TEMP PRICE'})
    
    df['TEMP PRICE'] = df['TEMP PRICE'].astype(str)

    
    df['TEMP PRICE'] = df['TEMP PRICE'].str.replace(',', '.', regex=False)
    
    df['TEMP PRICE'] = df['TEMP PRICE'].str.replace(r'[^0-9.]', '', regex=True)
    
    df['TEMP PRICE'] = df['TEMP PRICE'].astype(float)
    
    df = df[df['TEMP PRICE'] > 0]
    
    found_columns = []
    return df

def remove_non_temp_columns(df):
    temp_columns = [col for col in df.columns if 'TEMP' in col]
    
    df = df[temp_columns]
    
    return df

def clean_and_standardize_data(df):
    df = clean_column_names(df)
    df = standardize_date_columns(df)
    df = standardize_product_columns(df)    
    df = standardize_quantity_columns(df)
    df = standardize_price_columns(df)
    df = remove_non_temp_columns(df)
    return df




# EXTRACT DATA

def process_aggregated_data(df):
    required_columns = ['TEMP DATE', 'TEMP PRODUCT', 'TEMP PRICE', 'TEMP QUANTITY']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input DataFrame harus memiliki kolom {required_columns}")

    # Tambahkan informasi tanggal
    df['TEMP DATE'] = pd.to_datetime(df['TEMP DATE'])
    df['Year'] = df['TEMP DATE'].dt.year
    df['Month'] = df['TEMP DATE'].dt.month
    df['Day'] = df['TEMP DATE'].dt.day
    df['Is_Weekend'] = df['TEMP DATE'].dt.weekday >= 5  

    # Agregasi awal berdasarkan tanggal
    grouped = df.groupby('TEMP DATE')
    aggregated_data = grouped.agg(
        Total_Transactions=('TEMP DATE', 'size'),
        Total_Products=('TEMP PRODUCT', 'nunique'),
        Highest_Price=('TEMP PRICE', 'max'),
        Lowest_Price=('TEMP PRICE', 'min'),
        Total_Quantity=('TEMP QUANTITY', 'sum'),
        Mean_Quantity_Per_Transaction=('TEMP QUANTITY', 'mean'),
        Highest_Quantity_Product=('TEMP QUANTITY', 'max'),
        Lowest_Quantity_Product=('TEMP QUANTITY', 'min'),
        Profit_Per_Day=('TEMP PRICE', lambda x: (x * df.loc[x.index, 'TEMP QUANTITY']).sum())
    ).reset_index()

    # Fungsi untuk menghitung fitur tambahan
    def calculate_additional_features(group):
        # Agregasi berdasarkan produk untuk setiap tanggal
        temp_df = group.groupby('TEMP PRODUCT').agg(
            total_quantity=('TEMP QUANTITY', 'sum'),
            mean_price=('TEMP PRICE', 'mean')
        ).reset_index()

        # Urutkan berdasarkan harga dan quantity
        sorted_prices = temp_df.sort_values('mean_price')['mean_price'].unique()
        sorted_quantities = temp_df.sort_values('total_quantity')['total_quantity']

        # Harga tertinggi dan terendah
        highest_price = sorted_prices[-1] if len(sorted_prices) > 0 else 0
        lowest_price = sorted_prices[0] if len(sorted_prices) > 0 else 0

        # Harga kedua tertinggi dan terendah
        second_highest_price = sorted_prices[-2] if len(sorted_prices) > 1 else 0
        second_lowest_price = sorted_prices[1] if len(sorted_prices) > 1 else 0

        # Median price
        median_price = sorted_prices[(len(sorted_prices) - 1) // 2] if len(sorted_prices) > 0 else 0

        # Total quantity berdasarkan harga
        total_quantity_highest_price = temp_df[temp_df['mean_price'] == highest_price]['total_quantity'].sum()
        total_quantity_lowest_price = temp_df[temp_df['mean_price'] == lowest_price]['total_quantity'].sum()
        total_quantity_second_highest_price = temp_df[temp_df['mean_price'] == second_highest_price]['total_quantity'].sum()
        total_quantity_second_lowest_price = temp_df[temp_df['mean_price'] == second_lowest_price]['total_quantity'].sum()
        total_quantity_median_price = temp_df[temp_df['mean_price'] == median_price]['total_quantity'].sum()

        # Quantity tertinggi, terendah, dan median
        highest_quantity = sorted_quantities.iloc[-1] if len(sorted_quantities) > 0 else 0
        lowest_quantity = sorted_quantities.iloc[0] if len(sorted_quantities) > 0 else 0
        second_highest_quantity = sorted_quantities.iloc[-2] if len(sorted_quantities) > 1 else 0
        second_lowest_quantity = sorted_quantities.iloc[1] if len(sorted_quantities) > 1 else 0
        median_quantity = sorted_quantities.iloc[(len(sorted_quantities) - 1) // 2] if len(sorted_quantities) > 0 else 0

        # Harga berdasarkan quantity
        price_highest_quantity = temp_df[temp_df['total_quantity'] == highest_quantity]['mean_price'].iloc[0] if len(temp_df[temp_df['total_quantity'] == highest_quantity]) > 0 else 0
        price_lowest_quantity = temp_df[temp_df['total_quantity'] == lowest_quantity]['mean_price'].iloc[0] if len(temp_df[temp_df['total_quantity'] == lowest_quantity]) > 0 else 0
        price_second_highest_quantity = temp_df[temp_df['total_quantity'] == second_highest_quantity]['mean_price'].iloc[0] if len(temp_df[temp_df['total_quantity'] == second_highest_quantity]) > 0 else 0
        price_second_lowest_quantity = temp_df[temp_df['total_quantity'] == second_lowest_quantity]['mean_price'].iloc[0] if len(temp_df[temp_df['total_quantity'] == second_lowest_quantity]) > 0 else 0
        price_median_quantity = temp_df[temp_df['total_quantity'] == median_quantity]['mean_price'].iloc[0] if len(temp_df[temp_df['total_quantity'] == median_quantity]) > 0 else 0

        return pd.Series({
            'Second_Highest_Price': second_highest_price,
            'Second_Lowest_Price': second_lowest_price,
            'Median_Price': median_price,
            'Median_Quantity': median_quantity,
            'Total_Quantity_Highest_Price': total_quantity_highest_price,
            'Total_Quantity_Lowest_Price': total_quantity_lowest_price,
            'Total_Quantity_Second_Highest_Price': total_quantity_second_highest_price,
            'Total_Quantity_Second_Lowest_Price': total_quantity_second_lowest_price,
            'Total_Quantity_Median_Price': total_quantity_median_price,
            'Price_Highest_Quantity': price_highest_quantity,
            'Price_Lowest_Quantity': price_lowest_quantity,
            'Second_Highest_Quantity': second_highest_quantity,
            'Second_Lowest_Quantity': second_lowest_quantity,
            'Price_Second_Highest_Quantity': price_second_highest_quantity,
            'Price_Second_Lowest_Quantity': price_second_lowest_quantity,
            'Price_Median_Quantity': price_median_quantity
        })

    # Terapkan fungsi tambahan pada grup tanggal
    additional_features = df.groupby('TEMP DATE').apply(calculate_additional_features).reset_index()
    aggregated_data = pd.merge(aggregated_data, additional_features, on='TEMP DATE', how='left')

    # Tambahkan informasi tanggal kembali ke hasil agregasi
    aggregated_data['Year'] = aggregated_data['TEMP DATE'].dt.year
    aggregated_data['Month'] = aggregated_data['TEMP DATE'].dt.month
    aggregated_data['Day'] = aggregated_data['TEMP DATE'].dt.day
    aggregated_data['Is_Weekend'] = aggregated_data['TEMP DATE'].dt.weekday >= 5

    # Urutkan kolom sesuai keinginan
    desired_column_order = [
        'TEMP DATE', 'Year', 'Month', 'Day', 'Is_Weekend',
        'Total_Transactions', 'Total_Products', 'Highest_Price', 'Lowest_Price',
        'Second_Highest_Price', 'Second_Lowest_Price', 'Median_Price',
        'Total_Quantity', 'Mean_Quantity_Per_Transaction', 
        'Highest_Quantity_Product', 'Lowest_Quantity_Product',
        'Second_Highest_Quantity', 'Second_Lowest_Quantity', 'Median_Quantity',
        'Total_Quantity_Highest_Price', 'Total_Quantity_Lowest_Price',
        'Total_Quantity_Second_Highest_Price', 'Total_Quantity_Second_Lowest_Price',
        'Total_Quantity_Median_Price', 'Price_Highest_Quantity', 'Price_Lowest_Quantity',
        'Price_Second_Highest_Quantity', 'Price_Second_Lowest_Quantity', 'Price_Median_Quantity',
        'Profit_Per_Day'
    ]
    aggregated_data = aggregated_data[desired_column_order]
    aggregated_data = aggregated_data.rename(columns={'TEMP DATE': 'Date'})

    return aggregated_data

def SplittingData(df, split_index):
    
    return X_train, y_train, X_test, y_test


mainModel = XGBRegressor(
    n_estimators=10000,
    learning_rate=0.01,
    max_depth=12,    
    subsample=0.6,
    colsample_bytree=0.9,
)

def firstTrain(df, model, split_index, with_eval_set=True):
    
    return model

def predictSingleColumnTierOne(df):
    
    return model

def createForecast(df, numOfDay):
    
    return df


@app.post("/forecast")
async def predict(input_data: InputData):
    
    df = pd.DataFrame(input_data.query_result)

    
    df = clean_and_standardize_data(df)
    df = process_aggregated_data(df)

    
    df_forecast = createForecast(df, input_data.num_days)
    
    
    df_forecast['Profit_Per_Day'] = df_forecast['sales_quantity'] * df_forecast['price_per_item']
    result = df_forecast[['Date', 'Profit_Per_Day']].to_dict(orient='records')
    
    return {"forecast": result}

# Jalankan FastAPI dengan Uvicorn:
# uvicorn script_name:app --reload
