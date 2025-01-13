import pandas as pd

# File path to your CSV file
file_path_indre_data = "./CSV-Data/38.05_Vandstand_Minut.csv"
file_path_ydre_data = "./CSV-Data/Coastal Authority)_Minut.csv"
file_path_past_weather = "./CSV-Data/Hvide_Sande__5200 (1).csv"

with open(file_path_indre_data, "r") as file:
    lines_indre = file.readlines()

with open(file_path_ydre_data, "r") as file:
    lines_ydre = file.readlines()

# Step 2: Identify the data start index
indre_data_start_idx = next(i for i, line in enumerate(lines_indre) if line.strip() and line.strip()[0].isdigit())
ydre_data_start_idx = next(i for i, line in enumerate(lines_ydre) if line.strip() and line.strip()[0].isdigit())

try:
    indre_data = pd.read_csv(file_path_indre_data, skiprows=indre_data_start_idx, sep=';', engine='python')
    ydre_data = pd.read_csv(file_path_ydre_data, skiprows=ydre_data_start_idx, sep=';', engine='python')

    indre_data_array = indre_data.values.tolist()
    ydre_data_array = ydre_data.values.tolist()
    print("Data successfully loaded:")
    #print(indre_data.head(10))
    #print(ydre_data.head(10))

    #print(data_array)
except FileNotFoundError:
    print(f"Error: The file at '{file_path_indre_data}' was not found.")

except pd.errors.ParserError:
    print("Error: There was an issue parsing the file. Please check the file format.")
    
except Exception as e:
    print(f"An unexpected error occurred: {e}")


indre_vandstande = []
ydre_vandstande = []

for entry in indre_data_array:
    vandstand = entry[1]
    indre_vandstande.append(vandstand)

#print('her starter den anden')
for entry in ydre_data_array:
    vandstand = entry[1]
    ydre_vandstande.append(vandstand)
#print(len(indre_vandstande))
#print(len(ydre_vandstande))