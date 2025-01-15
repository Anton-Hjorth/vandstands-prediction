import pandas as pd
import datetime

def import_csv(file_path):
    """data_list = []
    
    with open(file_path, "r") as file:
        lines = file.readlines()

    try:
        valid_start = next(i for i, line in enumerate(lines) if line.strip() and line.strip()[0].isdigit())
        data = pd.read_csv(file_path, skiprows=valid_start, sep=';', engine='python')
        data_array = data.values.tolist()
        
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
        data_array = []

    except pd.errors.ParserError:
        print("Error: There was an issue parsing the file. Please check the file format.")
        data_array = []
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        data_array = []

    for entry in data_array:
        datetime_str = entry[0]
        # timestamp = time.mktime(datetime.datetime.strptime(datetime_str, "%d-%m-%Y %H:%M").timetuple())
        timestamp = datetime.datetime.strptime(datetime_str, "%d-%m-%Y %H:%M")
        formatted_datetime_str = timestamp.strftime("%d-%m-%Y %H:%M")

        wanted_data = entry[1]
        data_list.append([formatted_datetime_str, wanted_data])
        
        return data_list
        """
    
    # Find the first row with data
    with open(file_path, 'r') as file:
        lines = file.readlines()
        valid_start = next(i for i, line in enumerate(lines) if line.strip() and line.strip()[0].isdigit())

    # Read the CSV with pandas starting from line that contains the first data entry
    data_frame = pd.read_csv(file_path, sep=";", skiprows=valid_start, usecols=[0, 1], names=["Timestamp", "Water Level"])
    return data_frame
    
indre_vandstande = import_csv("CSV-Data/Indre_Vandstand - fra 2019.csv")
ydre_vandstande = import_csv("CSV-Data/Ydre_Vandstand - fra 2019.csv")
print(indre_vandstande)
print(ydre_vandstande)
