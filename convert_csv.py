import pandas as pd

def import_csv(file_path):
    # Find the first row with data
    with open(file_path, 'r') as file:
        lines = file.readlines()
        valid_start = next(i for i, line in enumerate(lines) if line.strip() and line.strip()[0].isdigit())

    # Read the CSV with pandas starting from line that contains the first data entry
    data_frame = pd.read_csv(file_path, sep=";", skiprows=valid_start, usecols=[0, 1], names=["Timestamp", "Water Level"])
    return data_frame
    
indre_vandstande = import_csv("CSV-Data/Indre_Vandstand - fra 2019.csv")
ydre_vandstande = import_csv("CSV-Data/Ydre_Vandstand - fra 2019.csv")