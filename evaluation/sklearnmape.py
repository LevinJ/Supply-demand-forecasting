from sklearn.utils import assert_all_finite
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred, district_num = 66, dateslot_num = 144): 
    """
    Use of this metric is not recommended; for illustration only. 
    See other regression metrics on sklearn docs:
      http://scikit-learn.org/stable/modules/classes.html#regression-metrics
    Use like any other metric
    >>> y_true = [3, -0.5, 2, 7]; y_pred = [2.5, -0.3, 2, 8]
    >>> mean_absolute_percentage_error(y_true, y_pred)
    Out[]: 24.791666666666668
    """
    y_true = np.asanyarray(y_true)
    y_pred = np.asanyarray(y_pred)
    assert_all_finite(y_true)
    assert_all_finite(y_pred)
    #Filter zero values in y_true
    sel = (y_true != 0)
    y_true = y_true[sel]
    y_pred = y_pred[sel]
    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)
    return np.abs((y_true - y_pred) / y_true.astype(np.float32)).sum()/float(district_num * dateslot_num)
#     return np.mean(np.abs((y_true - y_pred) / y_true.astype(np.float32)))


def mean_absolute_percentage_error_scoring(estimator, X, y):
    y_pred = estimator.predict(X)
    return -mean_absolute_percentage_error(y, y_pred)

if __name__ == "__main__":   
    y_true = [3, -0.5, 2, 0,7]
    y_pred = [2.5, -0.3, 2, 8,0]
    print mean_absolute_percentage_error(y_true, y_pred)