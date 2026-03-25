import pandas as pd

def pts_to_array(timestamp: pd.Timestamp) -> list[int]:
    return [timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, timestamp.second]
