from io_utils.mat_io import MatLoader
import pandas as pd
from matplotlib import pyplot as plt

def test_read_mat():
    mat_data = MatLoader.load('/home/transen5/Project/atm_rpc/039BE-20240501-20240515-dt5_struct.mat')
    assert mat_data.name == '039BE'
    assert mat_data.dt == pd.Timedelta(seconds=5)
    assert mat_data.data['Ex1'].start() == mat_data.start_time

    ch = mat_data.data['Ex1']
    assert mat_data.datetime_index.size == ch.cts.size
    assert ch.datetime_index().size == mat_data.datetime_index.size
