import pandas as pd


#define dataframe
house_price = pd.read_csv('../sample_datasets/house_price.csv')

#fill nan values with arbitrary value - pandas
house_price['GarageCond'] = house_price['GarageCond'].fillna('None')
house_price['GarageYrBlt'] = house_price['GarageYrBlt'].fillna(-1)
