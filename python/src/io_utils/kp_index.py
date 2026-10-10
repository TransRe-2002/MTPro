import logging
import json
import pandas as pd
import numpy as np
import os
import h5py
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
from base.app_state import load_app_state

logger = logging.getLogger(__name__)


def align_to_3hour_boundary(timestamp: pd.Timestamp, direction: str = 'floor') -> pd.Timestamp:
    if timestamp.tz is None:
        timestamp = timestamp.tz_localize('UTC')
    else:
        timestamp = timestamp.tz_convert('UTC')

    hour = timestamp.hour

    if direction == 'floor':
        aligned_hour = (hour // 3) * 3
    elif direction == 'ceil':
        if hour % 3 == 0 and timestamp.minute == 0 and timestamp.second == 0:
            aligned_hour = hour
        else:
            aligned_hour = ((hour // 3) + 1) * 3
            if aligned_hour >= 24:
                timestamp = timestamp + pd.Timedelta(days=1)
                aligned_hour = 0
    else:
        raise ValueError("direction must be 'floor' or 'ceil'")

    return timestamp.replace(hour=aligned_hour, minute=0, second=0, microsecond=0)


def fetch_kp(start: pd.Timestamp, end: pd.Timestamp, hdf5_path: str = 'kp.h5') -> pd.DataFrame | None:
    """
    获取Kp指数数据，返回含 ['Kp_datetime', 'Kp'] 两列的DataFrame。
    优先从本地h5读取，缺失部分从网络补充并写回h5；网络不可用时缺失点填NaN。

    Returns:
        pd.DataFrame，列为 ['Kp_datetime', 'Kp']，与原始fetch_kp返回格式一致
    """
    if hdf5_path == 'kp.h5':
        hdf5_path = load_app_state().kp_hdf5_path

    # ── 1. 时区统一 & 对齐到3小时边界 ─────────────────────────────
    if start.tz is None:
        start = start.tz_localize('UTC')
    else:
        start = start.tz_convert('UTC')

    if end.tz is None:
        end = end.tz_localize('UTC')
    else:
        end = end.tz_convert('UTC')

    aligned_start = align_to_3hour_boundary(start, direction='ceil')
    aligned_end   = align_to_3hour_boundary(end,   direction='floor')

    if aligned_start != start or aligned_end != end:
        logger.info("时间已对齐到3小时边界:")
        logger.info("  原始范围: %s 到 %s", start, end)
        logger.info("  调整后:   %s 到 %s", aligned_start, aligned_end)

    if aligned_start > aligned_end:
        logger.warning("对齐后起始时间晚于结束时间，无有效时间范围")
        return None

    # ── 2. 生成期望的完整3小时整点时间序列 ────────────────────────
    expected_times = pd.date_range(
        start=aligned_start,
        end=aligned_end,
        freq='3h',
        tz='UTC'
    )
    logger.info("期望时间点共 %s 个：%s 到 %s", len(expected_times), expected_times[0], expected_times[-1])

    # ── 3. 从本地h5加载已有数据（内部用index存储） ─────────────────
    local_df = _load_kp_internal(hdf5_path)  # index=Kp_datetime, col=Kp

    if local_df is not None and not local_df.empty:
        existing_times = set(local_df.index)
        missing_times  = [t for t in expected_times if t not in existing_times]
    else:
        local_df      = None
        missing_times = list(expected_times)

    logger.info(
        "本地已有 %s 个时间点，缺失 %s 个时间点",
        len(expected_times) - len(missing_times),
        len(missing_times),
    )

    # ── 4. 无缺失 → 直接从本地返回 ────────────────────────────────
    if not missing_times:
        result = local_df.loc[local_df.index.isin(expected_times)].sort_index()
        logger.info("本地数据完整，直接返回 %s 条记录", len(result))
        return _to_output_df(result)

    # ── 5. 有缺失 → 从第一个缺失点开始请求网络 ────────────────────
    fetch_start = missing_times[0]
    fetch_end   = expected_times[-1]
    logger.info("尝试从网络获取缺失数据：%s 到 %s", fetch_start, fetch_end)

    net_df = _fetch_kp_from_network(fetch_start, fetch_end)  # index=Kp_datetime, col=Kp

    # ── 6a. 网络成功 → 合并写回h5，返回结果 ───────────────────────
    if net_df is not None and not net_df.empty:
        frames = [df for df in [local_df, net_df] if df is not None and not df.empty]
        merged_df = pd.concat(frames)
        merged_df = merged_df[~merged_df.index.duplicated(keep='last')].sort_index()
        merged_df.index.name = 'Kp_datetime'

        _save_kp_internal(merged_df, hdf5_path)
        logger.info(
            "合并数据：本地 %s 条 + 网络 %s 条 = 合并后 %s 条",
            len(local_df) if local_df is not None else 0,
            len(net_df),
            len(merged_df),
        )

        result = merged_df.loc[merged_df.index.isin(expected_times)].sort_index()
        return _to_output_df(result)

    # ── 6b. 网络失败 → 缺失点填NaN ────────────────────────────────
    logger.warning("网络获取失败，将 %s 个缺失时间点填充为 NaN", len(missing_times))
    nan_df = pd.DataFrame(
        {'Kp': np.nan},
        index=pd.DatetimeIndex(missing_times, tz='UTC', name='Kp_datetime')
    )
    frames = [df for df in [local_df, nan_df] if df is not None and not df.empty]
    result_df = pd.concat(frames)
    result_df = result_df[~result_df.index.duplicated(keep='last')].sort_index()
    result_df.index.name = 'Kp_datetime'

    result = result_df.loc[result_df.index.isin(expected_times)].sort_index()
    return _to_output_df(result)


def _to_output_df(df: pd.DataFrame) -> pd.DataFrame:
    """将内部以index存储的DataFrame转为对外输出格式：['Kp_datetime', 'Kp'] 两列"""
    out = df.reset_index()
    out['Kp_datetime'] = out['Kp_datetime'].dt.tz_convert('UTC')
    return out[['Kp_datetime', 'Kp']]


def _fetch_kp_from_network(fetch_start: pd.Timestamp, fetch_end: pd.Timestamp) -> pd.DataFrame | None:
    """从GFZ网络获取Kp数据，返回以Kp_datetime为index的DataFrame，失败返回None"""
    try:
        params = urllib_parse.urlencode({
            'start': fetch_start.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'end': fetch_end.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'index': 'Kp',
            'status': 'all',
        })
        url = f"https://kp.gfz.de/app/json/?{params}"
        req = urllib_request.Request(
            url,
            headers={
                'Accept': 'application/json',
                'User-Agent': 'MTPro/1.0',
            },
            method='GET',
        )

        with urllib_request.urlopen(req, timeout=15) as response:
            status_code = getattr(response, 'status', response.getcode())
            if status_code != 200:
                logger.warning("GFZ Kp 接口返回非 200 状态码: %s", status_code)
                return None
            data = json.load(response)

        if not data:
            logger.warning("GFZ Kp 接口返回空响应")
            return None

        if isinstance(data, dict) and data.get('message'):
            logger.warning("GFZ Kp 接口返回错误消息: %s", data['message'])
            return None

        if not isinstance(data, dict) or 'datetime' not in data or 'Kp' not in data:
            logger.warning("GFZ Kp 接口返回格式异常")
            return None

        net_df = pd.DataFrame({
            'Kp_datetime': pd.to_datetime(data['datetime'], utc=True),
            'Kp': data['Kp']
        }).set_index('Kp_datetime').sort_index()
        return net_df
    except (urllib_error.HTTPError, urllib_error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as e:
        logger.warning("网络获取Kp数据失败: %s", e)
        return None


def _save_kp_internal(df: pd.DataFrame, hdf5_path: str):
    """保存以Kp_datetime为index的DataFrame到h5（h5py直写，无pytables依赖）"""
    if df is None or df.empty:
        return
    try:
        with h5py.File(hdf5_path, 'w') as f:
            grp = f.create_group('kp_data')
            grp.create_dataset(
                'axis0', data=np.array(['Kp_datetime'], dtype='S32'))
            # pandas HDF fixed格式: axis1 = int64 index (ns since epoch)
            ts_ns = df.index.asi8
            grp.create_dataset('axis1', data=ts_ns, dtype='i8')
            grp.create_dataset(
                'block0_items', data=np.array([df.columns[0]], dtype='S32'))
            grp.create_dataset(
                'block0_values', data=df.values.astype('f8'))
        logger.info("数据已保存到 %s，共 %s 条记录", hdf5_path, len(df))
    except Exception as e:
        logger.warning("保存数据时出错: %s", e)


def _load_kp_internal(hdf5_path: str) -> pd.DataFrame | None:
    """从h5读取数据（h5py直读，无pytables依赖），返回以Kp_datetime为index的DataFrame"""
    if not os.path.exists(hdf5_path):
        logger.info("文件 %s 不存在", hdf5_path)
        return None
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'kp_data' not in f:
                return None
            grp = f['kp_data']
            ts_ns = grp['axis1'][:]
            values = grp['block0_values'][:].ravel()
            col_name = grp['block0_items'][0]
            index_name = grp['axis0'][0]
            if isinstance(col_name, bytes):
                col_name = col_name.decode('utf-8')
            if isinstance(index_name, bytes):
                index_name = index_name.decode('utf-8')
        index = pd.DatetimeIndex(ts_ns, name='Kp_datetime').tz_localize('UTC')
        df = pd.DataFrame({'Kp': values.astype('f8')}, index=index)
        logger.info("从 %s 加载了 %s 条记录", hdf5_path, len(df))
        return df
    except Exception as e:
        logger.warning("加载数据时出错: %s", e)
        return None