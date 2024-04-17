import pandas as pd

def CheckColumnsPresent(data:pd.DataFrame, cols: list) -> bool:
    return all([col in data.columns for col in cols])