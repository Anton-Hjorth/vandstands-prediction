import requests
import pandas as pd
import datetime as dt

api_key = 'e318d53d-8ca4-47a9-b428-1eeb64536c7a'  # Insert your API key
DMI_URL = 'https://dmigw.govcloud.dk/v2/metObs/collections/observation/items'

# Specify the desired start and end time
start_time = pd.Timestamp(2019, 1, 1)
end_time = pd.Timestamp(2025, 1, 15)

# Convert to ISO format with UTC time zone
datetime_str = start_time.tz_localize('UTC').isoformat() + '/' + end_time.tz_localize('UTC').isoformat()

# Station IDs and parameter IDs for precipitation (rain data)
stationIds = ['06080'] # Esbjerg
parameterIds = ['precip_dur_past10min']  # Regn pr 10 min

# Prepare query parameters
params = {
    'api-key': api_key,
    'datetime': datetime_str,
    'stationId': ','.join(stationIds),
    'parameterId': ','.join(parameterIds),
    'limit': '300000',  # max limit
}

# Submit GET request with url and parameters
response = requests.get(DMI_URL, params=params)

if response.status_code == 200:
    json_data = response.json()  # Extract JSON data
    
    # Normalize JSON to a DataFrame
    df = pd.json_normalize(json_data['features'])

    # Ensure 'time' column is in datetime format
    df['time'] = pd.to_datetime(df['properties.observed'])

    # Clean the DataFrame by renaming and selecting relevant columns
    df = df[['time', 'properties.value', 'properties.stationId', 'properties.parameterId']]
    df.columns = df.columns.str.replace('properties.', '')  # Clean up column names
    
    # Drop duplicates based on 'time' and 'value'
    df = df.drop_duplicates(subset=['time', 'value'])

    # Set appropriate index
    df.set_index(['parameterId', 'stationId', 'time'], inplace=True)
    
    # Unstack the data to make it more readable
    df = df['value'].unstack(['stationId', 'parameterId'])

    # Write the DataFrame to a CSV file
    df.to_csv('precipitation_data.csv')

    print("Data has been written to 'precipitation_data.csv'")
else:
    print(f"Error: {response.status_code}")