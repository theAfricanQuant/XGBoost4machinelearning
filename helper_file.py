
import pandas as pd
import numpy as np
from wrangle_bike_rentals import *
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import model_selection

from sklearn.model_selection import cross_val_score

from category_encoders import OneHotEncoder


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


url_census = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

def prep_census(url):
    col_names = ['age', 'workclass', 'fnlwgt', 
                 'education', 'education-num', 
                 'marital-status', 'occupation',
                  'relationship', 'race', 'sex', 
                 'capital-gain', 'capital-loss', 
                 'hours-per-week', 'native-country', 
                   'income']
    return (pd
            .read_csv(url, header=None)
            .pipe(lambda x: x.rename(columns={i: name for i, name in enumerate(col_names)}))
            .drop(['education'], axis=1)
    )

df_census = prep_census(url_census)

def splitX_y(df, trgt):
    features = [col for col in df.columns if col not in trgt]
    return (df[features], df[trgt])

def prep_bike_data(data):
    return (data
            .assign(windspeed = data["windspeed"]
                    .fillna((data["windspeed"]
                             .median())),
                    hum = (data['hum']
                   .fillna(data.groupby('season')['hum']
                           .transform('median'))),
                    dteday = pd.to_datetime(data['dteday']),
                    mnth = lambda x: x['dteday'].dt.month,
                    yr = data['yr'].ffill()
                   )
            .drop(['dteday', 'casual','registered'], axis=1)
           )

url_bikes = 'https://raw.githubusercontent.com/theAfricanQuant/XGBoost4machinelearning/main/data/bike_rentals.csv'

class PrepDataTransformer(BaseEstimator,
    TransformerMixin):
    """
    This transformer takes a Pandas DataFrame containing our survey 
    data as input and returns a new version of the DataFrame. 
    
    ----------
    ycol : str, optional
        The name of the column to be used as the target variable.
        If not specified, the target variable will not be set.
    Attributes
    ----------
    ycol : str
        The name of the column to be used as the target variable.
    """
    def __init__(self, ycol=None):
        self.ycol = ycol
    
    def transform(self, X):
        return prep_bike_data(X)

    def fit(self, X, y=None):
        return self

df_bikes = get_data(url_bikes)

class OHETransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies one-hot encoding to columns of type 'object' 
    in a Pandas DataFrame. The one-hot encoding process converts categorical 
    columns into a format that can be provided to machine learning algorithms 
    to improve predictions.

    The transformer identifies columns with data type 'object' and uses the
    OneHotEncoder from the category_encoders library to perform the encoding.

    Attributes
    ----------
    cols : list
        List of column names in the DataFrame identified for one-hot encoding.
    
    encode : OneHotEncoder object
        The encoder instance from category_encoders that performs the actual 
        one-hot encoding.
    """

    def __init__(self):
        self.cols = None
        self.encode = None

    def fit(self, X, y=None):
        self.cols = X.select_dtypes(include=['object']).columns.tolist()
        self.encode = OneHotEncoder(cols=self.cols, use_cat_names=False)
        self.encode.fit(X[self.cols])
        return self

    def transform(self, X):
        X_encoded = self.encode.transform(X[self.cols])
        X = X.drop(columns=self.cols)
        return pd.concat([X, X_encoded], axis=1)

def grid_search_optim(df, trgt_vect, params, model):
    """
    This function performs hyperparameter optimization for a given model using GridSearchCV.
    
    Parameters:
    df (pd.DataFrame): The input data frame containing the features and target variable.
    trgt_vect (str): The name of the target variable column in the data frame.
    params (dict): The dictionary containing parameter grid for GridSearchCV.
    model (estimator object): The machine learning model instance to be optimized.
    
    Returns:
    None: This function prints the best parameters and the training and test scores but does not return any values.
    
    Usage:
    >>> optimize_model_parameters(dataframe, 'target_column_name', parameter_grid, DecisionTreeRegressor())
    
    Note:
    - The 'dataframe' should be a data frame containing at least a column with the name specified in 'trgt_vect', which is used as the target variable.
    - 'parameter_grid' should be a dictionary where keys are parameter names (as strings) and values are lists of parameter settings to try as values.
    - 'PrepDataTransformer' should be a predefined class or function to preprocess the data.
    """
    
    # Split the data into features and target variable
    df_X, df_y = splitX_y(df, trgt_vect)

    # Split the data into training and testing sets
    df_X_train, df_X_test, df_y_train, df_y_test = model_selection.train_test_split(
        df_X, df_y, test_size=0.2, random_state=43)
    
    # Define the pipeline with preprocessing steps and the model
    pipe = Pipeline([
        ('tweak', PrepDataTransformer()),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Fit and transform the training data and transform the test data using the pipeline
    X_train = pipe.fit_transform(df_X_train)
    X_test = pipe.transform(df_X_test)
    
    # Instantiate GridSearchCV to optimize the model parameters
    grid_model = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    
    # Fit the model with the optimized parameters
    grid_model.fit(X_train, df_y_train)

    # Extract the best parameters found by GridSearchCV
    best_params = grid_model.best_params_

    # Print the best parameters
    print(f"Best params: {best_params}")
    
    # Compute and print the training score (RMSE)
    best_score = np.sqrt(-grid_model.best_score_)
    print(f"Training score: {best_score:.3f}")

    # Predict the target variable on the test set
    y_pred = grid_model.predict(X_test)

    # Compute and print the test score (RMSE)
    rmse_test = mean_squared_error(df_y_test, y_pred)**0.5
    print(f'Test score: {rmse_test:.3f}')


