from io_utils.mat_io import MatSaver, MatLoader
from core.em_data import EMData
from core.mat_data import MatEMData

def test_save_mat():
    ts = MatLoader.load("/home/transen5/Project/atm_rpc/039BE-20240501-20240515-dt5_struct.mat")
    MatSaver.save(ts, "test1.mat")
    ts.__class__ = EMData
    MatSaver.save(ts, "test2.mat")

