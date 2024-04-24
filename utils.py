import pandas as pd

def CheckColumnsPresent(data:pd.DataFrame, cols: list) -> bool:
    return all([col in data.columns for col in cols])

def AllElementsAreThere(elements: list, there: list) -> bool:
    return all([e in there for e in elements])

def ThereAreDoubles(arr):
    return len(arr) != len(set(arr))

def MissingColumn(data:pd.DataFrame, cols: list) -> list:
    return [col for col in cols if col not in data.columns]