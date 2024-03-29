from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd

datestrs = ['1/3/2021','3/2/2021','3/3/2021','3/4/2021','3/5/2021']
dates = pd.to_datetime(datestrs)
print(dates)
print("======================================")

ts = Series([1, np.nan, np.nan, 8, 10], index = dates)
print(ts)

# 보간법 (시계열에서 좋음)
ts_intp_linear = ts.interpolate()
print(ts_intp_linear)
