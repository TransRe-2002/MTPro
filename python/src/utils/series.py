import numpy as np
from pandas import DatetimeIndex, Series


def safe_nanmean(values) -> float:
    """不发警告的 nanmean：空数组 / 全 NaN 返回 nan，由调用方兜底。

    新版 numpy 中 ``nanmean`` 对空切片经 ``warnings.warn`` 抛
    "Mean of empty slice" RuntimeWarning，``np.errstate`` 拦不住
    （errstate 只管浮点错误信号），故先显式判空再调用。
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.isnan(arr).all():
        return float('nan')
    return float(np.nanmean(arr))


def dti_to_numpy(dti):
    if isinstance(dti, DatetimeIndex):
        # asi8 返回的是 DatetimeIndex 原生分辨率的整数值（可能是 ns/us/ms），
        # 统一转为纳秒后再除以 1e9，兼容 pandas 不同版本的默认分辨率。
        return dti.as_unit('ns').asi8.astype(np.float64) / 1e9
    if isinstance(dti, Series):
        return dti.dt.tz_convert('UTC').values.astype('datetime64[ns]').astype(np.float64) / 1e9
    return np.asarray(dti, dtype=np.float64)
