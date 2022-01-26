import pandas as pd
from feature_engine.imputation import DropMissingData

#define dataframe
house_price = pd.read_csv('../sample_datasets/house_price.csv')

#delete by rows
house_price = house_price.dropna(axis=0)
house_price = house_price.dropna(axis='rows')

#delete by columns
house_price = house_price.dropna(axis=1)
house_price = house_price.dropna(axis='columns')

#delete with permanently
house_price.dropna(axis=1, inplace=True)

#delete by row's and column's nan count
house_price = house_price.dropna(how='all')
house_price = house_price.dropna(how='any')
house_price = house_price.dropna(how='any', axis=1)

#delete by row's and column's thresh non-NA count
house_price = house_price.dropna(thresh=75)
house_price = house_price.dropna(thresh=75, axis=1)

#delete by only selected columns
house_price = house_price.dropna(subset=['PoolQC'])

#delete specific column
house_price = house_price.drop(['PoolQC'], axis=1)
house_price = house_price.drop(columns=['PoolQC'])
house_price.drop(columns=['PoolQC'], inplace=True)

#delete specific row
house_price = house_price.drop(index=[0, 1])
house_price.drop(index=[0, 1], inplace=True)

#delete by rows only selected columns - feature_engine
imputer = DropMissingData(variables=['LotFrontage', 'MasVnrArea'])
house_price = imputer.fit_transform(house_price)

#delete by row's thresh percentage non-NA count - feature_engine
imputer = DropMissingData(variables=['LotFrontage', 'MasVnrArea'], threshold=0.5)
house_price = imputer.fit_transform(house_price)
