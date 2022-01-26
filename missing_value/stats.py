import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from feature_engine.imputation import MeanMedianImputer
from feature_engine.imputation import CategoricalImputer


#define dataframe
house_price = pd.read_csv('../sample_datasets/house_price.csv')

#fill nan values with mean value - pandas
house_price['LotFrontage'] = house_price['LotFrontage'].fillna(house_price['LotFrontage'].mean())

#fill nan values with median value - pandas
house_price['LotFrontage'] = house_price['LotFrontage'].fillna(house_price['LotFrontage'].median())

#fill nan values with mode value - pandas
house_price['GarageYrBlt'] = house_price['GarageYrBlt'].fillna(house_price['GarageYrBlt'].mode()[0])


#fill nan values with mean value - sklearn
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
house_price['LotFrontage'] = imputer.fit_transform(house_price['LotFrontage'].values.reshape(len(house_price['LotFrontage'].values), 1))

#fill nan values with median value - sklearn
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
house_price['LotFrontage'] = imputer.fit_transform(house_price['LotFrontage'].values.reshape(len(house_price['LotFrontage'].values), 1))

#fill nan values with mode value - sklearn
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
house_price['GarageYrBlt'] = imputer.fit_transform(house_price['GarageYrBlt'].values.reshape(len(house_price['GarageYrBlt'].values), 1))


#fill nan values with mean value - feature_engine
imputer = MeanMedianImputer(imputation_method='mean',
                            variables=['LotFrontage', 'MasVnrArea'])
imputer.fit(house_price)
imputer.variables_
imputer.imputer_dict_
house_price = imputer.transform(house_price)

#fill nan values with median value - feature_engine
imputer = MeanMedianImputer(imputation_method='median',
                            variables=['LotFrontage', 'MasVnrArea'])
house_price = imputer.fit_transform(house_price)

#fill nan values with mode value - feature_engine
imputer = CategoricalImputer(imputation_method='frequent',
                             variables=['BsmtQual'])
house_price = imputer.fit_transform(house_price)