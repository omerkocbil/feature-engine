import pandas as pd


#define dataframe
house_price = pd.read_csv('../sample_datasets/house_price.csv')

#fill nan values with mean value - pandas
house_price['LotFrontage'] = house_price['LotFrontage'].fillna(house_price['LotFrontage'].mean())

#fill nan values with median value - pandas
house_price['LotFrontage'] = house_price['LotFrontage'].fillna(house_price['LotFrontage'].median())

#fill nan values with mode value - pandas
house_price['GarageYrBlt'] = house_price['GarageYrBlt'].fillna(house_price['GarageYrBlt'].mode()[0])
