import numpy as np
import matplotlib.pyplot as plt
from convert_csv import indre_vandstande, ydre_vandstande

# Extract water level data
indre_levels = np.array(indre_vandstande["Water Level"])
ydre_levels = np.array(ydre_vandstande["Water Level"])

# Ensure both arrays have the same length
min_length = min(len(indre_levels), len(ydre_levels), 1000)
indre_levels = indre_levels[:min_length]
ydre_levels = ydre_levels[:min_length]

# Create an x-axis based on the length of the data
x = np.arange(min_length)

plt.plot(x, indre_levels, '-', label='Indre Vandstand')
plt.plot(x, ydre_levels, '-', label='Ydre Vandstand')

# Find intersection points
idx = np.argwhere(np.diff(np.sign(indre_levels - ydre_levels))).flatten()

# Interpolate to find the y-values at the intersection points
intersect_x = []
intersect_y = []
for i in idx:
    x1, x2 = x[i], x[i+1]
    y1, y2 = indre_levels[i], indre_levels[i+1]
    y3, y4 = ydre_levels[i], ydre_levels[i+1]
    
    # Linear interpolation
    intersect_x_val = x1 + (x2 - x1) * (y1 - y3) / ((y1 - y3) - (y2 - y4))
    intersect_y_val = y1 + (y2 - y1) * (intersect_x_val - x1) / (x2 - x1)
    
    intersect_x.append(intersect_x_val)
    intersect_y.append(intersect_y_val)

print("Intersection points (x, y):", list(zip(intersect_x, intersect_y)))
plt.plot(intersect_x, intersect_y, 'ro', label='Intersections')

plt.xlabel('Index')
plt.ylabel('Water Level')
plt.legend()
plt.show()