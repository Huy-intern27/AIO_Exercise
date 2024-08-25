import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.dates as mdates # type: ignore

# Import library and read dataset
data_path = '/content/opsd_germany_daily.csv'

df = pd.read_csv(data_path)
print(df.shape)
print(df.dtypes)
print(df.head(3))

df.set_index('Date', inplace=True)
print(df.head(3))

opsd_daily = pd.read_csv('/content/opsd_germany_daily.csv', index_col='Date', parse_dates=True)
opsd_daily['Year'] = opsd_daily.index.year
opsd_daily['Month'] = opsd_daily.index.month
opsd_daily['Day'] = opsd_daily.index.day_name()
print(opsd_daily.head())

# Time-based Indexing
print(opsd_daily['2014-01-20':'2014-01-22'])
print(opsd_daily.loc['2012-02'])

# Visualizing time series data
## Plot one column
sns.set(rc={'figure.figsize': (11, 4)})
opsd_daily['Consumption'].plot(linewidth=0.5)

## Plot columns
cols_plot = ['Consumption', 'Solar', 'Wind']
axes = opsd_daily[cols_plot].plot(marker='.', figsize=(11, 9), subplots=True, alpha=0.5, linestyle='None')

for ax in axes:
  ax.set_ylabel('Daily Totals (Gwh)')

plt.show()

# Seasonality
fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
for name, ax in zip(['Consumption', 'Solar', 'Wind'], axes):
  sns.boxplot(data=opsd_daily, x='Month', y=name, ax=ax)
  ax.set_ylabel('GWh')
  ax.set_title(name)

  if ax != axes[-1]:
    ax.set_xlabel('')
    
# Frequencies
print(pd.date_range('1998-03-10', '1998-03-15', freq='D'))

time_sample =['2013-02-03', '2013-02-06', '2013-02-08']
consum_sample = opsd_daily.loc[time_sample, ['Consumption']].copy()
print(consum_sample)

consum_freq = consum_sample.asfreq('D')
consum_freq['Consumption Frequencies'] = consum_sample.asfreq('D', method='ffill')
print(consum_freq)

# Resampling
data_columns = ['Consumption', 'Wind', 'Solar', 'Wind+Solar']
opsd_weekly_mean = opsd_daily[data_columns].resample('W').mean()
print(opsd_weekly_mean.head())

print(opsd_daily.shape[0])
print(opsd_weekly_mean.shape[0])

## Visualize daily and weekly time in 6 months
start, end = '2017-01', '2017-06'

fig, ax = plt.subplots()
ax.plot(opsd_daily.loc[start:end, 'Solar'], marker='.', linestyle='-', linewidth=0.5, label='Daily')
ax.plot(opsd_weekly_mean.loc[start:end, 'Solar'], marker='o', markersize=8, linestyle='-', label='Weekly Mean Resample')
ax.set_ylabel('Solar Consumption (GWh)')
ax.legend()
plt.show()

## Visualize product of wind and solar over consumption
opsd_annual = opsd_daily[data_columns].resample('Y').sum(min_count=360)
opsd_annual = opsd_annual.set_index(opsd_annual.index.year)
opsd_annual.index.name = 'Year'
opsd_annual['Wind+Solar/ Consumption'] = opsd_annual['Wind+Solar'] / opsd_annual['Consumption']
print(opsd_annual)

ax = opsd_annual.loc[2012:, 'Wind+Solar/ Consumption'].plot.bar(color='C0')
ax.set_ylabel('Faction')
ax.set_ylim(0, 0.3)
ax.set_title('Wind + Solar Share of Annual Electricity Consumption')
plt.xticks(rotation=0)

# Rolling windows 
opsd_7d = opsd_daily[data_columns].rolling(7, center=True).mean()
print(opsd_7d.head(10))

# Trends
opsd_365d = opsd_daily[data_columns].rolling(window=365, center=True, min_periods=360).mean()

fig, ax = plt.subplots()
ax.plot(opsd_daily['Consumption'], marker='.', markersize=2, color='0.6', linestyle='None', label='Daily')
ax.plot(opsd_7d['Consumption'], linewidth=2, label='7-d Rolling Mean')
ax.plot(opsd_365d['Consumption'], color='0.2', linewidth=3, label='Trend (365d - Rolling Mean)')

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.legend()
ax.set_xlabel('Year')
ax.set_ylabel('Consumption (GWh)')
ax.set_title('Trends in Electricity Consumption')
plt.show()

fig, ax = plt.subplots()
for nm in ['Wind', 'Solar', 'Wind+Solar']:
  ax.plot(opsd_365d[nm], label=nm)
  ax.xaxis.set_major_locator(mdates.YearLocator())
  ax.set_ylim(0, 400)
  ax.legend()
  ax.set_ylabel('Consumption (GWh)')
  ax.set_title('Trends in Electricity Production (365-d Rolling Mean)')

plt.show()