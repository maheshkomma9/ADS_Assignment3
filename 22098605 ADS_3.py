#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:40:16 2024

@author: maheshroyal
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split




#Read and clean the datafile
def read_file(fn):
    """
    Reads a CSV file and returns a pandas DataFrame.

    Parameters:
    ------------    
    fn (str): The filename of the CSV file to be read.

    Returns:
    ---------    
    df (pandas.DatFrame): The DataFrame containing the data 
    read from the CSV file.
    """
    address = "/Users/maheshroyal/Downloads" + fn
    df = pd.read_csv(address)
    df = df.drop(df.columns[:2], axis=1)
    df=df.drop(columns=['Country Code'])
    
    # Remove the string of year from column names
    df.columns = df.columns.str.replace(' \[YR\d{4}\]', '', regex=True)
    countries=['Japan','India']
    country_code=['JPN','IND']
    
    #Transpose the dataframe
    df = df[df['Country Name'].isin(countries)].T 
    #Rename columns
    df = df.rename({'Country Name': 'year'})
    df = df.reset_index().rename(columns={'index': 'year'})
    
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    df = df.replace('..', np.nan)
    df = df.replace(np.nan,0)
    
    df["year"] = df["year"].astype(int)
    df["India"]=df["India"].astype(float)
    df["Japan"]=df["Japan"].astype(float)
    
    return df

''' cluster plot'''
def kmeans_scatter_plot(data1, data2, start_year, end_year):
    # Extract relevant columns
    columns_to_select = ['Country Name'] + [str(year) for year in range(start_year, end_year + 1)]
    df1 = data1[columns_to_select].copy()
    df2 = data2[columns_to_select].copy()

    # Merge the two dataframes based on 'Country Name'
    merged_df = pd.merge(df1, df2, on='Country Name', suffixes=('_data1', '_data2'))

    # Drop rows with missing values
    merged_df.dropna(inplace=True)

    # Prepare data for clustering
    X = merged_df.iloc[:, 1:]  # Select all columns except 'Country Name'

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    merged_df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Plot the clustered data
    plt.figure(figsize=(10, 8))
    plt.scatter(merged_df.iloc[:, 1], merged_df.iloc[:, 2], c=merged_df['Cluster'], cmap='viridis', edgecolors='k', s=50)
    plt.xlabel('HIV Prevelence',color='grey',fontsize=20)
    plt.ylabel('Population', color='grey',fontsize=20)
    plt.title('Population vs HIV Prevalence (1990-2022)', color='lightcoral',fontsize='20')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

# Load your datasets
data1 = pd.read_excel('API_SH.HIV.INCD.TL.P3_DS2_en_excel_v2_6299550.xls',skiprows=3)  # Replace with the actual path
data2 = pd.read_excel('API_SP.POP.TOTL_DS2_en_excel_v2_6299418.xls',skiprows=3)  # Replace with the actual path

# Specify the range of years for columns
start_year = 1990
end_year = 2020

# Create the scatter plot with K-means clustering
kmeans_scatter_plot(data1, data2, start_year, end_year)


''' Actual data plotting of HIV Incidence'''

def read_file(file_name):
    my_data_set = pd.read_excel(file_name,skiprows=3)
    countries=['Colombia']
    
    #retreving required data
    req_set_of_years=[str(year) for year in range(1990, 2022)]
    my_req_set_countries=my_data_set[my_data_set['Country Name'].isin(countries)]
    [['Country Name'] + req_set_of_years]
    
    print(my_req_set_countries.columns)
    #plot the figure
    plt.figure(figsize=(12, 6))
    for index, row in my_req_set_countries.iterrows():
        plt.plot(req_set_of_years, row[req_set_of_years], label=row['Country Name'])
    


    # Adding labels and title
    plt.xlabel('Year', color='grey',fontsize=20)
    plt.ylabel('Population in Billions',fontsize=20,color='deepskyblue')
    plt.title('Hiv prevrelance Rate',fontsize=20, color='mediumblue')
    plt.legend()
    plt.xticks(req_set_of_years[::3], fontsize='15')
    plt.yticks(fontsize=15)
    plt.savefig('22098605.png',dpi=300)
file_name = 'API_SH.HIV.INCD.TL.P3_DS2_en_excel_v2_6299550.xls'
read_file(file_name)

''' Actual data plotting of population'''

def read_file(file_name):
    my_data_set = pd.read_excel(file_name,skiprows=3)
    countries=['Colombia']
    
    #retreving required data
    req_set_of_years=[str(year) for year in range(1990, 2022)]
    my_req_set_countries2=my_data_set[my_data_set['Country Name'].isin(countries)]
    [['Country Name'] + req_set_of_years]
    
    #plot the figure
    plt.figure(figsize=(12, 6))
    for index, row in my_req_set_countries2.iterrows():
        plt.plot(req_set_of_years, row[req_set_of_years], label=row['Country Name'])
    


    # Adding labels and title
    plt.xlabel('Year',color='grey',fontsize=20)
    plt.ylabel('Population in Billions',fontsize=20,color='deepskyblue')
    plt.title('population Rate',fontsize=20, color='mediumblue')
    plt.legend()
    plt.xticks(req_set_of_years[::3], fontsize='15')
    plt.yticks(fontsize=15)
    plt.savefig('22098605.png',dpi=300)
file_name = 'API_SP.POP.TOTL_DS2_en_excel_v2_6299418.xls'
read_file(file_name)


#prediction values of colombia population
def read_file(file_name):
    

    # Read the dataset

    my_data_set = pd.read_excel(file_name, skiprows=3)
    print(my_data_set)
    # Select China's population data from the dataset
    colombia_population = my_data_set[my_data_set['Country Name'] == 'Colombia'][['Country Name'] + [str(year) for year in range(1970, 2022)]]

    # Prepare the data for curve fitting
    years = np.array(range(1970, 2022))  # Use data up to 2022 for fitting
    population_values = colombia_population.values[:, 1:].flatten()

    # Define a polynomial function for curve fitting
    def polynomial_curve(x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x + d

    # Use curve_fit to fit the data
    params, covariance = curve_fit(polynomial_curve, years, population_values)

    # Make predictions for the years 1970 to 2030
    years_extended = np.array(range(1970, 2031))  # Extend the range to 2030
    predicted_population_1970_2030 = polynomial_curve(years_extended, *params)

    # Plot the results
    plt.figure(figsize=(12, 10))
    plt.plot(years, population_values, label='Actual Data')
    plt.plot(years_extended, predicted_population_1970_2030, color='orange', label='Curve Fitting Model')
    plt.scatter([2030], predicted_population_1970_2030[-1], color='blue', marker='X', label='Predicted for 2030')
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Population in Billions', fontsize=20, color='deepskyblue')
    plt.title('Colombia Population Prediction (Curve Fitting)', fontsize=20, color='mediumblue')
    plt.legend()
    plt.grid(linestyle='--', linewidth=0.5, color='black')
    plt.xticks(range(1970, 2031, 5), fontsize='15')
    plt.yticks(fontsize=15)
    plt.savefig('colombia_population_prediction_curve_fit_2030.png', dpi=300)
    plt.show()

    # Display the predicted population for 2030
    print(f'Predicted Population for China in 2030: {predicted_population_1970_2030[-1]:,.2f} Billion')
file_name='API_SP.POP.TOTL_DS2_en_excel_v2_6299418.xls'
read_file(file_name)


''' prediction of hiv incidence rate in colombia'''
# Read the dataset
def read_file(file_name): 
    my_data_set = pd.read_excel(file_name, skiprows=3)

    # Select China's population data from the dataset
    colombia_hiv_prevalance = my_data_set[my_data_set['Country Name'] == 'Colombia'][['Country Name'] + [str(year) for year in range(1990, 2022)]]

    # Prepare the data for curve fitting
    years = np.array(range(1990, 2022))  # Use data up to 2022 for fitting
    population_values = colombia_hiv_prevalance.values[:, 1:].flatten()

    # Define a polynomial function for curve fitting
    def polynomial_curve(x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x + d

    # Use curve_fit to fit the data
    params, covariance = curve_fit(polynomial_curve, years, population_values)

    # Make predictions for the years 1970 to 2030
    years_extended = np.array(range(1990, 2031))  # Extend the range to 2030
    predicted_population_1970_2030 = polynomial_curve(years_extended, *params)

    # Plot the results
    plt.figure(figsize=(12, 10))
    plt.plot(years, population_values, label='Actual Data')
    plt.plot(years_extended, predicted_population_1970_2030, color='red', label='Curve Fitting Model')
    plt.scatter([2030], predicted_population_1970_2030[-1], color='green', marker='X', label='Predicted for 2030')
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('HIV Incidence Rate', fontsize=20, color='deepskyblue')
    plt.title('HIV prevalence  Prediction (Curve Fitting)', fontsize=20, color='mediumblue')
    plt.legend()
    plt.grid(linestyle='--', linewidth=0.5, color='black')
    plt.xticks(range(1990, 2031, 5), fontsize='17')
    plt.yticks(fontsize=15)
    plt.savefig('china_population_prediction_curve_fit_2030.png', dpi=300)
    plt.show()

    # Display the predicted population for 2030
    print(f'Predicted Population for China in 2030: {predicted_population_1970_2030[-1]:,.2f} Billion')
file_name = 'API_SH.HIV.INCD.TL.P3_DS2_en_excel_v2_6299550.xls'
read_file(file_name)