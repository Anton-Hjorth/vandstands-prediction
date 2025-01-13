import pandas as pd

# File path to your CSV file
file_path = "./CSV-Data/38.05_Vandstand_Minut.csv"

# Read the CSV file using pandas
# Since the data is separated by semicolons, specify the delimiter as ';'
try:
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines_to_skip = 0
        for i, line in enumerate(lines[:40]):
            print(line.strip()[0])
            if line.strip()[0] == '#':
                lines_to_skip =+ 1
                continue
        print(lines_to_skip)
        data = pd.read_csv(file_path, delimiter=';',skiprows=14)

    # Display the first few rows of the DataFrame to verify the import
    print("Data successfully loaded:")
    print(data.head(10))
except FileNotFoundError:
    print(f"Error: The file at '{file_path}' was not found.")
except pd.errors.ParserError:
    print("Error: There was an issue parsing the file. Please check the file format.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Example: Accessing the data for further processing
# for index, row in data.iterrows():
#     print(f"Iteration {index + 1}: {row.values}")

# Optional: Save as another format after processing
# data.to_csv("output.csv", index=False)
