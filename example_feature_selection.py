"""
Example of feature selection with Univariate Feature Selection wrapper.
"""
import pandas as pd
from sklearn.datasets import fetch_california_housing
from src.classes.UnivariateFeatureSelection import UnivariateFeatureSelection

# example dataset
df = pd.read_csv('data/housing/housing.csv')
target = 'median_house_value'
excludes = ['ocean_proximity','total_bedrooms']
features = [col for col in df.columns if col not in [target] and col not in excludes ]

X = df[features].values
y = df[target].values

univariate_fs = UnivariateFeatureSelection(
    n_features=4,
    problem_type='regression',
    scoring='f_regression'
)

univariate_fs.fit(X, y)
X_transformed = univariate_fs.transform(X)

print(X.shape)
print(X_transformed.shape)