import numpy as np
from pandas import DatetimeIndex, Series


def dti_to_numpy(dti):
    if isinstance(dti, DatetimeIndex):
        # asi8 返回的是 DatetimeIndex 原生分辨率的整数值（可能是 ns/us/ms），
        # 统一转为纳秒后再除以 1e9，兼容 pandas 不同版本的默认分辨率。
        return dti.as_unit('ns').asi8.astype(np.float64) / 1e9
    if isinstance(dti, Series):
        return dti.dt.tz_convert('UTC').values.astype('datetime64[ns]').astype(np.float64) / 1e9
    return np.asarray(dti, dtype=np.float64)
