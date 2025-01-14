import numpy as np
import matplotlib.pyplot as plt
from convert_csv import indre_vandstande, ydre_vandstande, wather_data

array1 = indre_vandstande
array2 = ydre_vandstande

# test = [[x, y], [3, 4], [5, 6], [7, 8], [9, 10]]

# Extract x and y values directly using list comprehensions
array1_x_list = [item[0] for item in array1]
array1_y_list = [item[1] for item in array1]

array2_x_list = [item[0] for item in array2]
array2_y_list = [item[1] for item in array2]

# Interpolate array2_x_list to match the length of array1_x_list
x1_interp = np.linspace(min(array1_x_list), max(array1_x_list), num=len(array1_x_list))
y1_interp = np.interp(x1_interp, np.linspace(min(array1_x_list), max(array1_x_list), num=len(array2_x_list)), array2_x_list)
y1_interp = [round(float(val), 0) for val in y1_interp]

# Interpolate array2_y_list to match the length of array1_y_list
x2_interp = np.linspace(min(array1_y_list), max(array1_y_list), num=len(array1_y_list))
y2_interp = np.interp(x2_interp, np.linspace(min(array1_y_list), max(array1_y_list), num=len(array2_y_list)), array2_y_list)

# Combine the interpolated values into array2
array2_interp = [[round(float(x), 2), round(float(y), 2)] for x, y in zip(y1_interp, y2_interp)]


#print(array2_interp)

"""
array2_interp_x = [item[0] for item in array2_interp]
array2_interp_y = [item[1] for item in array2_interp]
# Plot the interpolated array2
plt.plot(array2_interp_x, array2_interp_y, label="Interpolated Array 2")
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Interpolated Array 2')
plt.legend()
plt.show()"""









"""
simplificer scriptet
få det til at passe med [timestamp, vindstyrke, retning]
"""