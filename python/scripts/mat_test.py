from scipy.io import loadmat

mat = loadmat("/home/transen5/Project/atm_rpc/039BE-20240501-20240515-dt5_struct.mat")
print(mat['tsStruct']['startTime'][0][0][0])
print(mat['tsStruct']['npts'][0][0][0][0])
print(mat['tsStruct']['npts'][0, 0][0, 0])
print(mat['tsStruct']['chid'][0][0][0])
print(mat['tsStruct']['chid'][0][0][0][0])
print(mat['tsStruct']['chid'][0, 0][0, 0][0])
print(mat['tsStruct']['chid'][0, 0][0][0][0])
print(mat['tsStruct']['UserData'][0][0][0])
print(mat['tsStruct']['UserData'][0][0][0][0])
print(mat['tsStruct']['clockRef'][0][0])
print(mat['tsStruct']['latitude'][0][0][0][0])
print(mat['tsStruct']['longitude'][0, 0][0, 0])
print(mat['tsStruct']['name'][0, 0][0])
print(mat['tsStruct']['name'][0, 0][0][0])
print(mat['tsStruct']['dt'][0, 0][0, 0])
if (arr := mat['tsStruct']['elevation'][0, 0]).size == 0:
    print(arr)
else:
    print(arr[0])
print(mat['tsStruct']['units'][0, 0])
print(mat['tsStruct']['units'][0, 0][0])
print(mat['tsStruct']['units'][0, 0][0, :][0])
print(mat['tsStruct']['units'][0, 0][0, :][0][0])
print(mat['tsStruct']['units'][0, 0][0, 0])
print(mat['tsStruct']['units'][0, 0][0, 0][0])

