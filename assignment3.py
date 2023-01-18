#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 23:40:26 2023

@author: bilalnazir
"""

import numpy as np
import pandas as pd

def get_data_frames(filename,countries,indicator):
    '''
    This function returns two dataframes one with countries as column and other one years as column.
    It tanspose the dataframe and converts rows into column and column into rows of specific column and rows.
    It takes three arguments defined as below. 

    Parameters
    ----------
    filename : Text
        Name of the file to read data.
    countries : List
        List of countries to filter the data.
    indicator : Text
        Indicator Code to filter the data.

    Returns
    -------
    df_countries : DATAFRAME
        This dataframe contains countries in rows and years as column.
    df_years : DATAFRAME
        This dataframe contains years in rows and countries as column..

    '''
    # Read data using pandas in a dataframe.
    df = pd.read_csv(filename, skiprows=(4), index_col=False)
    # Get datafarme information.
    df.info()
    # To clean data we need to remove unnamed column.
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # To filter data by countries
    df = df.loc[df['Country Name'].isin(countries)]
    # To filter data by indicator code.
    df = df.loc[df['Indicator Code'].eq(indicator)]
    
    # Using melt function to convert all the years column into rows as one column
    df2 = df.melt(id_vars=['Country Name','Country Code','Indicator Name','Indicator Code'], var_name='Years')
    # Deleting country code column.
    del df2['Country Code']
    # Using pivot table function to convert countries from rows to separate column for each country.   
    df2 = df2.pivot_table('value',['Years','Indicator Name','Indicator Code'],'Country Name').reset_index()
    
    df_countries = df
    df_years = df2
    
    # Cleaning data droping nan values.
    df_countries.dropna()
    df_years.dropna()
    
    return df_countries, df_years


# List of countries 
countries = ['Germany','Australia','United States','China','United Kingdom']
# calling functions to get dataframes and use for plotting graphs.
df_c, df_y = get_data_frames('API_19_DS2_en_csv_v2_4700503.csv',countries,'AG.LND.FRST.ZS')

df_c.dropna()

print(df_c['Country Name'])




