import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import FunctionTransformer

import warnings
warnings.filterwarnings('ignore')


#define dataframe
titanic = pd.read_csv('../../sample_datasets/titanic.csv', usecols=['Age','Fare','Survived'])

## TODO: Bu bölüme eda kütüphanemden aynı işi yapan fonksiyonu koy
#draw dist graph and q-q plot - age
plt.figure(figsize=(15,6))

plt.subplot(121)
sns.distplot(titanic['Age'])
plt.title('Age PDF')

plt.subplot(122)
stats.probplot(titanic['Age'], dist="norm", plot=plt)
plt.title('Age QQ Plot')

#draw dist graph and q-q plot - fare
plt.figure(figsize=(15,6))

plt.subplot(121)
sns.distplot(titanic['Fare'])
plt.title('Age PDF')

plt.subplot(122)
stats.probplot(titanic['Fare'], dist="norm", plot=plt)
plt.title('Age QQ Plot')

#log transformations - sklearn
transformer = FunctionTransformer(func=np.log1p)
titanic['Fare'] = transformer.fit_transform(titanic['Fare'])