import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from forcast import indre_array, ydre_array

def get_intersection(start_line, end_line, indre_array, ydre_array):
    # Extract water level data and timestamps
    indre_levels = indre_array[start_line:end_line, 1].astype(float)
    ydre_levels = ydre_array[start_line:end_line, 1].astype(float)
    timestamps = indre_array[start_line:end_line, 0]  # Assuming both have the same timestamps

    # Convert timestamps to datetime64 objects
    x = np.array([np.datetime64(pd.to_datetime(ts, format="%d-%m-%Y %H:%M")) for ts in timestamps])

    plt.figure(figsize=(10, 6))
    plt.plot(x, indre_levels, '-', label='Indre Vandstand')
    plt.plot(x, ydre_levels, '-', label='Ydre Vandstand')

    # Find intersection points
    idx = np.argwhere(np.diff(np.sign(indre_levels - ydre_levels))).flatten()

    # Interpolate to find the y-values at the intersection points
    intersect_x = []
    intersect_y = []
    for i in idx:
        if i + 1 < len(x):  # Ensure the index is within the valid range
            x1, x2 = x[i], x[i+1]
            y1, y2 = indre_levels[i], indre_levels[i+1]
            y3, y4 = ydre_levels[i], ydre_levels[i+1]
            
            # Linear interpolation
            intersect_x_val = x1 + (x2 - x1) * (y1 - y3) / ((y1 - y3) - (y2 - y4))
            intersect_y_val = y1 + (y2 - y1) * (intersect_x_val - x1) / (x2 - x1)
            
            intersect_x.append(intersect_x_val)
            intersect_y.append(float(intersect_y_val))  # Convert np.float64 to Python float

    # Format the intersection points to include only the date and time
    formatted_intersections = [(pd.to_datetime(x_val).strftime('%Y-%m-%d %H:%M'), y_val) for x_val, y_val in zip(intersect_x, intersect_y)]
    
    plt.plot(intersect_x, intersect_y, 'ro', label='Intersections')

    plt.xlabel('Timestamp')
    plt.ylabel('Water Level')
    plt.legend()
    plt.savefig("intersection.png")
    # plt.show()

    return formatted_intersections

intersection = get_intersection(0, 48, indre_array, ydre_array) # start, end line from csv file "144 = half a day" "288 = a day"

# Create a dictionary with the intersection points
intersection_dict = {timestamp: water_level for timestamp, water_level in intersection}
# print(intersection_dict)