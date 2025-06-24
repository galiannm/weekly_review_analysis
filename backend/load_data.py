import json
import pandas as pd

# Load reviews and convert to DataFrame
def load_data(file_name):
    with open(f"data/{file_name}.json", "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.to_period("W")
    return df