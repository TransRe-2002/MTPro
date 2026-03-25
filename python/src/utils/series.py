import numpy as np

from pandas import DatetimeIndex

def dti_to_numpy(dti: DatetimeIndex) -> np.ndarray:
    return np.array([ts.timestamp() for ts in dti])
