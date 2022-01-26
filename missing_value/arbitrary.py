import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


#define dataframe
house_price = pd.read_csv('../sample_datasets/house_price.csv')

#fill nan values with arbitrary value - pandas
house_price['GarageCond'] = house_price['GarageCond'].fillna('None')
house_price['GarageYrBlt'] = house_price['GarageYrBlt'].fillna(-1)

#fill nan values with arbitrary value - sklearn
imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1)
house_price['GarageYrBlt'] = imputer.fit_transform(house_price['GarageYrBlt'].values.reshape(len(house_price['GarageYrBlt'].values), 1))
