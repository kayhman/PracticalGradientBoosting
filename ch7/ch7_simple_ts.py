import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create time serie timestamp indices
monthly_variation = [0.0, 0.0, 0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.4, 0.2, 0.05, 0.0]
weekly_sales = [20, 25, 30, 25, 20, 30, 0]
constant_sales = [10, 10, 10, 10, 10, 10, 10]


X = pd.DataFrame({'date': pd.date_range(start='1/1/2021', end='30/12/2021'),
                  'value': np.array(weekly_sales * 52) + np.random.normal(0, 2, 364)})

X['day_of_week'] = X['date'].apply(lambda date: date.day_of_week)
X['month'] = X['date'].apply(lambda date: date.month)

plt.style.use('grayscale')
plt.plot(X[X.day_of_week == 0]['date'], X[X.day_of_week == 0]['value'], label='Lundi')
plt.plot(X[X.day_of_week == 1]['date'], X[X.day_of_week == 1]['value'], label='Mardi')
plt.plot(X[X.day_of_week == 2]['date'], X[X.day_of_week == 2]['value'], label='Mercredi')
plt.plot(X[X.day_of_week == 3]['date'], X[X.day_of_week == 3]['value'], label='Jeudi')
plt.plot(X[X.day_of_week == 4]['date'], X[X.day_of_week == 4]['value'], label='Vendredi')
plt.plot(X[X.day_of_week == 5]['date'], X[X.day_of_week == 5]['value'], label='Samedi')
plt.plot(X[X.day_of_week == 6]['date'], X[X.day_of_week == 6]['value'], label='Dimanche')
plt.legend(loc='upper left')

plt.savefig('day_of_week.png')
plt.show()

# Smoothed time serie
X['smoothed'] = X['value'].ewm(span=7*4).mean()

plt.ylim(0, 30)
plt.style.use('grayscale')
plt.plot(X['date'], X['smoothed'], label='Série temporelle Lissée')
plt.legend()

plt.savefig('smoothed.png')
plt.show()

# Raw time serie
plt.style.use('grayscale')
plt.plot(X['date'].iloc[:7*6], X['value'].iloc[:7*6], label='Série temporelle')
plt.legend()

plt.savefig('timeserie.png')
plt.show()
