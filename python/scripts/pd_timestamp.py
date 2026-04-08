import pandas as pd
import pytz

data = pd.Timestamp.now(tz=pytz.timezone("Asia/Shanghai"))
array = data.to_pydatetime()
print(array)