import pandas as pd

# File path to your CSV file
file_path_indre_data = "./CSV-Data/38.05_Vandstand_Minut.csv"
file_path_ydre_data = "./CSV-Data/Coastal Authority)_Minut.csv"
file_path_past_weather = "./CSV-Data/Hvide_Sande_Weather.csv"

with open(file_path_indre_data, "r") as file:
    lines_indre = file.readlines()

with open(file_path_ydre_data, "r") as file:
    lines_ydre = file.readlines()

with open(file_path_past_weather, "r") as file:
    lines_wather = file.readlines()

# Step 2: Identify the data start index
indre_data_start_idx = next(i for i, line in enumerate(lines_indre) if line.strip() and line.strip()[0].isdigit())
ydre_data_start_idx = next(i for i, line in enumerate(lines_ydre) if line.strip() and line.strip()[0].isdigit())
weather_data_start_idx = next(i for i, line in enumerate(lines_wather) if line.strip() and line.strip()[0].isdigit())

try:
    indre_data = pd.read_csv(file_path_indre_data, skiprows=indre_data_start_idx, sep=';', engine='python')
    ydre_data = pd.read_csv(file_path_ydre_data, skiprows=ydre_data_start_idx, sep=';', engine='python')
    weather_data = pd.read_csv(file_path_past_weather, skiprows=weather_data_start_idx, sep=',', engine='python')
    
    indre_data_array = indre_data.values.tolist()
    ydre_data_array = ydre_data.values.tolist()
    weather_data_array = weather_data.values.tolist()

except FileNotFoundError:
    print(f"Error: The file at '{file_path_indre_data}' was not found.")

except pd.errors.ParserError:
    print("Error: There was an issue parsing the file. Please check the file format.")
    
except Exception as e:
    print(f"An unexpected error occurred: {e}")


indre_vandstande = []
ydre_vandstande = []
wather_data = []

for entry in indre_data_array:
    vandstand = entry[1]
    indre_vandstande.append(vandstand)

#print('her starter den anden')
for entry in ydre_data_array:
    vandstand = entry[1]
    ydre_vandstande.append(vandstand)

for entry in weather_data_array:
    wind_speed = entry[1]
    wind_direction = entry[2]
    gust_wind = entry[5]
    wather_data.append([wind_direction, wind_speed, gust_wind])

# print(len(indre_vandstande))
# print(len(ydre_vandstande))
# print(wather_data)