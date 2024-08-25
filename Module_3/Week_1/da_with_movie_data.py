import numpy as np # type: ignore
import pandas as pd # type: ignore

# Read data
df = pd.read_csv('data/IMDB-Movie-Data.csv')
df_indexed = pd.read_csv('data/IMDB-Movie-Data.csv', index_col='Title')

# View data
print(df.head())

# Understand some basic information about the data
print(df.info())
print(df.describe())

# Data Selection - Indexing and Slicing data
genre = df['Genre']
print(genre)

some_cols = df[['Title', 'Genre', 'Actors', 'Director', 'Rating']]
print(some_cols)

df_slicing = df[10:15][['Title', 'Genre', 'Actors', 'Director', 'Rating']]
print(df_slicing)

# Data Selection - Based on Conditional filtering
df_conditions = df[((df['Year'] >= 2010) & (df['Year'] <= 2015) & (df['Rating'] < 6.0) & (df['Revenue (Millions)'] > df['Revenue (Millions)'].quantile(0.95)))]
print(df_conditions)

# Groupby Opearations
df_groupby = df.groupby('Director')[['Rating']].mean()
print(df_groupby)

# Sorting Operations
df_sort = df.groupby('Director')[['Rating']].mean().sort_values(by='Rating',ascending=False).head()
print(df_sort)

# View missing value
missing = df.isnull().sum()
print(missing)

# Deal with missing values - Deleting
dropped_col = df.drop('Revenue (Millions)', axis=1).head()
print(dropped_col)

drop_cols = df.dropna()
print(drop_cols)

# Dealing with missing values - Filling
revenue_mean = df_indexed['Revenue (Millions)'].mean()
df_indexed['Revenue (Millions)'].fillna(revenue_mean, inplace=True)
print(df.isnull.sum())

df_indexed['Metascore'].interpolate(inplace=True)
print(df.isnull.sum())

# apply() functions
def rating_group(rating):
  if rating >= 8.0:
    return 'Good'
  elif rating >= 7.0:
    return 'Average'
  else:
    return 'Bad'

df_indexed = df_indexed['Rating'].apply(rating_group)
print(df_indexed.head())

