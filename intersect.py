import numpy as np
import matplotlib.pyplot as plt
from convert_csv import indre_vandstande, ydre_vandstande
import pandas as pd

def get_intersection(start_line, end_line):
    # Extract water level data and timestamps
    indre_levels = np.array(indre_vandstande["Water Level"])
    ydre_levels = np.array(ydre_vandstande["Water Level"])
    timestamps = pd.to_datetime(indre_vandstande["Timestamp"], format="%d-%m-%Y %H:%M")  # Assuming both have the same timestamps

    # Ensure both arrays have the same length and select the desired range
    indre_levels = indre_levels[start_line:end_line]
    ydre_levels = ydre_levels[start_line:end_line]
    timestamps = timestamps[start_line:end_line]

    # Convert timestamps to numpy array
    x = np.array(timestamps)

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
            intersect_y.append(intersect_y_val)

    # Format the intersection points to include only the date and time
    formatted_intersections = [pd.to_datetime(x_val).strftime('%Y-%m-%d %H:%M:%S') for x_val in intersect_x]
    
    """plt.plot(intersect_x, intersect_y, 'ro', label='Intersections')

    plt.xlabel('Timestamp')
    plt.ylabel('Water Level')
    plt.legend()
    plt.show()"""

    return formatted_intersections

intersection = get_intersection(0, 288) # start, end line from csv file "144 = half a day" "288 = a day"
print(intersection) 