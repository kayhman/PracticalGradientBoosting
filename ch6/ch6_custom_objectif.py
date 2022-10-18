import pandas as pd
import numpy as np
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt


# log cosh quantile is a regularized quantile loss function
def log_cosh_quantile(alpha):
    def _log_cosh_quantile(y_true, y_pred):
        err = y_pred - y_true
        err = np.where(err < 0, alpha * err, (1 - alpha) * err)
        grad = np.tanh(err)
        hess = 1 / np.cosh(err)**2
        return grad, hess
    return _log_cosh_quantile

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

dtf = pd.read_csv('data/taxi_data.csv',
                  parse_dates=['tpep_dropoff_datetime', 'tpep_pickup_datetime'],
                  date_parser=dateparse)

# Compute trip duration in minutes
dtf['duration'] = (dtf['tpep_dropoff_datetime'] - dtf['tpep_pickup_datetime']).apply(lambda x : x.total_seconds() / 60.0)

# Do some minimal cleaning : remove outliers
dtf = dtf[(dtf['duration'] < 90) & (dtf['duration'] > 0)]

# identify useless columns and drop them
to_drop = ['tpep_dropoff_datetime',
           'tpep_pickup_datetime',
           'store_and_fwd_flag',
           'passenger_count',
           'RatecodeID',
           'store_and_fwd_flag',
           'PULocationID',
           'DOLocationID',
           'payment_type',
           'fare_amount',
           'extra',
           'mta_tax',
           'tip_amount']
dtf.drop(to_drop, axis=1, inplace=True)


# Create an object to split input dataset into train and test dataset
splitter = ShuffleSplit(n_splits=1, test_size=.25, random_state=0)

alpha = 0.95
to_predict = 'duration'

for train_index, test_index in splitter.split(dtf):
    train = dtf.iloc[train_index]
    test = dtf.iloc[test_index]

    X = train
    y = train[to_predict]
    X.drop([to_predict], axis=1, inplace=True)

    X_test = test
    y_test = test[to_predict]
    X_test.drop([to_predict], axis=1, inplace=True)

    # over predict
    clf = XGBRegressor(objective=log_cosh_quantile(alpha),
                       n_estimators=125,
                       max_depth=5,
                       n_jobs=6,
                       learning_rate=.05)

    clf.fit(X, y)
    y_upper_smooth = clf.predict(X_test)

    # under predict
    clf = XGBRegressor(objective=log_cosh_quantile(1-alpha),
                       n_estimators=125,
                       max_depth=5,
                       n_jobs=6,
                       learning_rate=.05)

    clf.fit(X, y)
    y_lower_smooth = clf.predict(X_test)
    res = pd.DataFrame({'lower_bound' : y_lower_smooth, 'true_duration': y_test, 'upper_bound': y_upper_smooth})

    max_length = 150
    fig = plt.figure()
    plt.plot(list(y_test[:max_length]), 'gx', label=u'Valeur réelle')
    plt.plot(y_upper_smooth[:max_length], 'y_', label=u'Q supérieur')
    plt.plot(y_lower_smooth[:max_length], 'b_', label=u'Q inférieur')
    index = np.array(range(0, len(y_upper_smooth[:max_length])))
    plt.fill(np.concatenate([index, index[::-1]]),
             np.concatenate([y_upper_smooth[:max_length], y_lower_smooth[:max_length][::-1]]),
             alpha=.5, fc='b', ec='None', label='Intervalle de confiance à 90% ')
    plt.xlabel('$index$')
    plt.ylabel('$duration$')
    plt.legend(loc='upper left')
    plt.style.use('grayscale')
    plt.savefig('confidence.png')
    plt.show()


    count = res[(res.true_duration >= res.lower_bound) & (res.true_duration <= res.upper_bound)].shape[0]
    total = res.shape[0]
