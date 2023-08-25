
import pandas as pd
import datetime as dt


def show_nulls(df):
    return (df[df
            .isna()
            .any(axis=1)]
           )
    
def total_nulls(df):
    return (df
         .isna()
         .sum()
         .sum()
        )

def get_data(url):
    return (
        pd.read_csv(url)
    )

def mean_vals(df, idx1, idx2, col):
    return (
        (df.iloc[idx1][col] + 
        df.iloc[idx2][col])/2
    )

def prep_data(data):
    return (data
            .assign(windspeed = data["windspeed"]
                    .fillna((data["windspeed"]
                             .median())),
                    hum = (data['hum']
                   .fillna(data.groupby('season')['hum']
                           .transform('median'))),
                    temp = (data['temp']
                            .fillna(mean_vals(data, 700, 702, 'temp'))),
                    atemp = (data['atemp']
                            .fillna(mean_vals(data, 700, 702, 'atemp'))),
                    dteday = pd.to_datetime(data['dteday']),
                    mnth = lambda x: x['dteday'].dt.month,
                    yr = data['yr'].ffill()
                   )
            .drop(['dteday', 'casual','registered'], axis=1)
           )
