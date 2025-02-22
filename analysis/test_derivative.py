import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.stats import linregress

def identify_glass_transition(data):
    temperature = data[:, 1]
    density = data[:, 4]

    # Step 1: Use K-means clustering to identify different segments of data
    X = np.vstack((temperature, density)).T
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    labels = kmeans.labels_

    # Step 2: Analyze each cluster for linearity
    linear_clusters = []
    for cluster_label in np.unique(labels):
        cluster_temp = temperature[labels == cluster_label]
        cluster_density = density[labels == cluster_label]
        slope, intercept, r_value, p_value, std_err = linregress(cluster_temp, cluster_density)
        predicted = slope * cluster_temp + intercept
        error = np.mean((cluster_density - predicted) ** 2)
        linear_clusters.append((cluster_label, error, cluster_temp, cluster_density))

    # Step 3: Sort clusters by temperature to ensure selection from opposite ends
    linear_clusters.sort(key=lambda x: x[2].mean())

    # Select one linear region from the low-temperature end and one from the high-temperature end
    region1_temp, region1_density = linear_clusters[0][2], linear_clusters[0][3]
    region2_temp, region2_density = linear_clusters[-1][2], linear_clusters[-1][3]

    # Limit the high-temperature region to exclude temperatures above 430K
    region2_mask = (region2_temp < 445) & (region2_temp > 350)
    region2_temp = region2_temp[region2_mask]
    region2_density = region2_density[region2_mask]

    # Step 4: Fit robust linear models to both regions using RANSAC
    model1 = RANSACRegressor(LinearRegression()).fit(region1_temp.reshape(-1, 1), region1_density)
    model2 = RANSACRegressor(LinearRegression()).fit(region2_temp.reshape(-1, 1), region2_density)

    # Step 5: Find the intercept of the two lines
    slope1 = model1.estimator_.coef_[0]
    intercept1 = model1.estimator_.intercept_
    slope2 = model2.estimator_.coef_[0]
    intercept2 = model2.estimator_.intercept_

    # Solve for temperature where the two lines intersect
    if slope1 != slope2:
        intersection_temp = (intercept2 - intercept1) / (slope1 - slope2)
        intersection_density = slope1 * intersection_temp + intercept1
    else:
        intersection_temp = np.nan  # Parallel lines, no intersection
        intersection_density = np.nan

    # Plotting the results
    #plt.scatter(temperature, density, s=10, color='gray', alpha=0.5)
    #plt.plot(region1_temp, model1.predict(region1_temp.reshape(-1, 1)), color='blue', label='Fit Region 1')
    #plt.plot(region2_temp, model2.predict(region2_temp.reshape(-1, 1)), color='green', label='Fit Region 2')
    #plt.axvline(x=intersection_temp, color='red', linestyle='--', label=f'Tg = {intersection_temp:.2f} K')
    #plt.xlabel('Temperature (K)')
    #plt.ylabel('Density (kg/m^3)')
    #plt.legend()
    #plt.show()

    return intersection_temp, intersection_density, slope1, slope2, intercept1, intercept2, model1, model2, region1_temp, region2_temp

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.stats import linregress
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

def linear_fit(x, a, b):
    return a * x + b

def tanh_fit(x, a, b, c, d):
    return a * np.tanh(b * (x - c)) + d

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def sliding_window_slope(temperature, density, window_size):
    slopes = []
    mid_points = []
    half_window = window_size // 2

    for i in range(half_window, len(temperature) - half_window):
        temp_window = temperature[i - half_window:i + half_window + 1]
        dens_window = density[i - half_window:i + half_window + 1]
        slope, _, _, _, _ = linregress(temp_window, dens_window)
        slopes.append(slope)
        mid_points.append(temperature[i])

    return np.array(mid_points), np.array(slopes)

def plot_slope_temperature(data, window_size=50, smooth=False, sigma=1):
    temperature = data[:, 1]
    density = data[:, 4]

    # Sort the data by temperature to ensure correct calculation
    sorted_indices = np.argsort(temperature)
    temperature = temperature[sorted_indices]
    density = density[sorted_indices]

    # Apply sliding window linear regression to calculate the slope
    temperature_mid, slope = sliding_window_slope(temperature, density, window_size)

    # Apply Gaussian smoothing if specified
    if smooth:
        slope = gaussian_filter1d(slope, sigma=sigma)

    # Calculate the second derivative (change in slope) to identify inflection points
    second_derivative = np.gradient(slope, temperature_mid)
    if smooth:
        second_derivative = gaussian_filter1d(second_derivative, sigma=sigma)

    # Plot the slope vs temperature
    plt.figure(figsize=(10, 5))
    plt.plot(temperature_mid, slope, color='#FF45CC', label='Slope (dDensity/dT)')
    plt.ylabel(r"Slope (kg/K $\cdot$ m$^3$)", fontsize=12)
    plt.xlabel('Temperature (K)', fontsize=12)
    plt.grid(True, which='both', axis='y', linestyle='-', linewidth=0.8, color='lightgray')  # Match gridline color and thickness
    plt.grid(True, which='major', axis='x', linestyle='-', linewidth=0.8, color='lightgray')  # Match x-axis gridlines as well
    plt.title('Slope of Density vs Temperature')
    #plt.legend()
    plt.show()

# Example usage:
import os
os.chdir(r'D:\pythongraphs\results_pdms\pdms28200')
data = np.genfromtxt('energycr0001.xvg', skip_header=27)
#identify_glass_transition(data)
plot_slope_temperature(data, window_size=650, smooth=True, sigma=2)
