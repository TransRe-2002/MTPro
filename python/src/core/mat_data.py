import numpy as np
import pandas as pd
import pytz

from core.em_data import EMData, Channel
from utils.fetch_kp import fetch_kp
from utils.timestamp import pts_to_array

class MatEMData(EMData):
    # mat文件读取来的数据，特有字段
    mat_meta: dict
    user_data: list[str]

    def __init__(self, mat_meta: dict, path: str):
        super().__init__(path)
        self.mat_meta = mat_meta
        tsStruct = mat_meta['tsStruct']

        self.name: str = tsStruct['name'][0, 0][0]
        self.dt = pd.Timedelta(seconds=tsStruct['dt'][0, 0][0, 0])
        self.npts = tsStruct['npts'][0, 0][0, 0]
        self.NCh = tsStruct['NCh'][0, 0][0, 0]
        zt = tsStruct['zeroTime'][0, 0][0]
        st = tsStruct['startTime'][0, 0][0]
        et = tsStruct['endTime'][0, 0][0]

        self.zero_time = pd.Timestamp(zt[0], zt[1], zt[2], zt[3],
            zt[4], zt[5], tz=pytz.timezone('Asia/Shanghai'))
        self.start_time = pd.Timestamp(st[0], st[1], st[2], st[3],
            st[4], st[5], tz=pytz.timezone('Asia/Shanghai'))
        self.end_time = pd.Timestamp(et[0], et[1], et[2], et[3],
            et[4], et[5], tz=pytz.timezone('Asia/Shanghai'))
        self.datetime_index = pd.date_range(
            start=self.start_time,
            end=self.end_time,
            freq=self.dt,
        )

        self.latitude = tsStruct['latitude'][0, 0][0, 0]
        self.longitude = tsStruct['longitude'][0, 0][0, 0]
        if (arr := tsStruct['elevation'][0, 0]).size != 0:
            self.elevation = arr[0]
        else:
            self.elevation = None

        data = tsStruct['data'][0, 0]
        chid = tsStruct['chid'][0, 0][0]
        for i in range(len(chid)):
            ch_name = chid[i][0]
            self.chid.append(ch_name)
            self.data[ch_name] = Channel(
                name=ch_name,
                cts=pd.Series(data[i, :]),
                parent=self,
            )

        units = None
        if (arr := tsStruct['units'][0, 0]).size != 0:
            units = arr[0]
        if units is not None and units.size == 2:
            self.e_units = units[1][0]
            self.m_units = units[0][0]
        elif units is not None and units.size == 1:
            ch_name: str = self.chid[0][0]
            if ch_name.startswith('E'):
                self.e_units = units[0][0]
                self.m_units = None
            elif ch_name[0] in ['B', 'H']:
                self.e_units = None
                self.m_units = units[0][0]
        else:
            self.e_units = None
            self.m_units = None

        self.kp_data = fetch_kp(
            self.start_time,
            self.end_time)

    def update_meta(self):
        # 更新原始数据结构中的字段
        tsStruct = self.mat_meta['tsStruct']
        data = np.empty((self.NCh, self.npts), dtype=np.float64)
        for i, channel in enumerate(self.chid):
            if channel in self.data:
                data[i, :] = self.data[channel].cts.to_numpy()

        tsStruct['data'][0, 0] = data
        tsStruct['startTime'][0, 0] = np.array([pts_to_array(self.start_time)])
        tsStruct['endTime'][0, 0] = np.array([pts_to_array(self.end_time)])

        tsStruct['dt'][0, 0] = np.array([self.dt.total_seconds()])
        tsStruct['npts'][0, 0] = np.array([[self.npts]])
        tsStruct['NCh'][0, 0] = np.array([[self.NCh]])