import matplotlib.pyplot as plt
import numpy as np
import random
import csv


vessels = 3
nr_timewindows =2
nr_turbines=5# 
speed_knots = 21 #arbitrary speed
price_disel = 2 # pris per nm utgår nå per liter, kan byttes tilbake 
price_eletric = 0.075 #dollar per kwh
max_nm_battery =50  #nm,  is constant speed so just arbiratry distance it covers on battery
type_technichans = 2 #how many types
nr_of_technichnas = (24,24) #index per type
min_time_charger = 15 #min not in use
nr_chargers=2
fleets = 3

min_distance_km = 2
nr_of_technichnas2d = []
for t in range(nr_timewindows):
    nr_of_technichnas2d.append(nr_of_technichnas)


# Define coordinates for the wind farm area and the base node at Haugesund Harbour
coordinates = [
    (4.300644, 59.357497),  # Point 1
    (4.267506, 59.447506),  # Point 2
    (4.556085, 59.472124),  # Point 3
    (4.591884, 59.382323),  # Point 4
]
base_node = (5.248061, 59.422539)  # Coordinates for KYLLINGØY


# Function definitions: in_polygon, generate_random_point_within_polygon, degrees_to_km

# Function to check if a point is inside the given polygon
def in_polygon(x, y, poly):
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n+1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# Function to generate a random point within the polygon that meets the minimum distance requirement
def generate_random_point_within_polygon(polygon, min_dist, existing_points):
    min_x, min_y = np.min(polygon, axis=0)
    max_x, max_y = np.max(polygon, axis=0)
    while True:
        random_point = (random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if in_polygon(random_point[0], random_point[1], polygon) and all(
            np.linalg.norm(np.array(random_point) - np.array(point)) >= min_dist for point in existing_points
        ):
            return random_point
        
turbines = []
while len(turbines) < nr_turbines:
    point = generate_random_point_within_polygon(coordinates, min_distance_km / 111, turbines)  # Convert km to degrees
    turbines.append(point)

# Function to generate random points (now including charging stations)
def generate_random_point_within_polygon_with_exclusion(polygon, min_dist, existing_points, num_points=1):
    points = []
    min_x, min_y = np.min(polygon, axis=0)
    max_x, max_y = np.max(polygon, axis=0)
    while len(points) < num_points:
        random_point = (random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if in_polygon(random_point[0], random_point[1], polygon) and all(
            np.linalg.norm(np.array(random_point) - np.array(point)) >= min_dist for point in existing_points + points
        ):
            points.append(random_point)
    return points

# Generate 2 charging stations



# Function to convert degrees to kilometers (Haversine formula)
def degrees_to_km(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def km_to_nm(km):
    """Convert kilometers to nautical miles."""
    return km / 1.852



# def generate_maintenance_revenues(num_turbines):
#     """Generate random maintenance revenue for each turbine."""
#     return np.random.uniform(4000, 20000, size=num_turbines)


def generate_maintenance_revenues_with_time_windows(num_turbines, num_time_windows):
    #print(num_time_windows)
    revenues = np.random.uniform(1000, 4000, size=num_turbines*num_time_windows)
    window = []
    for t in range(1,num_time_windows+1):
        for i in range(1,num_turbines+1):
            window.append(t) 
    time_windows = np.random.randint(1, num_time_windows + 1, size=num_turbines*num_time_windows)
    #print(list(zip(revenues, time_windows)))
    return list(zip(revenues, window))




def generate_technician_demand(num_turbines, type_technicians):
    technician_demand = []
    for _ in range(num_turbines):
        # Generate the total number of technicians needed for the job
        total_technicians_needed = random.randint(4, 8)
        
        # Assign technician types randomly
        technician_types = np.random.randint(1, type_technicians + 1, total_technicians_needed)
        
        # Count the occurrence of each technician type
        technician_counts = [np.count_nonzero(technician_types == i + 1) for i in range(type_technicians)]
        
        technician_demand.append(technician_counts)
    
    return technician_demand

# Function to generate random maintenance times for each turbine
def generate_maintenance_times(num_turbines):
    return np.random.uniform(1, 6, size=num_turbines)

charging_stations = generate_random_point_within_polygon_with_exclusion(coordinates, min_distance_km / 111, turbines, nr_chargers)



# Generate turbines and calculate the distance matrix including the base node


turbines_with_base = [base_node] + turbines  +charging_stations # Include base node for distance calculations
turbine_coords = np.array(turbines_with_base)
distance_matrix_km = np.zeros((len(turbine_coords), len(turbine_coords)))
for i in range(len(turbine_coords)):
    for j in range(i + 1, len(turbine_coords)):
        distance = degrees_to_km(turbine_coords[i][1], turbine_coords[i][0], turbine_coords[j][1], turbine_coords[j][0])
        distance_nm = km_to_nm(distance)  # Convert distance to nautical miles directly
        distance_matrix_km[i, j] = distance_matrix_km[j, i] = distance_nm

#Plotting turbines and the base node


# # Calculate cost matrices for diesel and electric
# cost_matrix_diesel = distance_matrix_km *price_disel
# print (cost_matrix_diesel)  # $50 per nm for diesel
# print("\n")
# cost_matrix_electric = distance_matrix_km *price_eletric # $25 per nm for electric
# print(cost_matrix_electric)
# battery_usage_matrix = (distance_matrix_km / max_nm_battery) * 100  # Convert distances to percentage of max_nm_battery

# distance_matrix_km = np.round(distance_matrix_km, 1)



# # Applying rounding to the cost and battery usage matrices
# cost_matrix_diesel = np.round(cost_matrix_diesel, 1)
# cost_matrix_electric = np.round(cost_matrix_electric, 1)
# battery_usage_matrix = np.round(battery_usage_matrix, 1)


technician_demand = generate_technician_demand(nr_turbines, type_technichans)


# maintenance_revenues = generate_maintenance_revenues(nr_turbines)

# time_matrix_hours = np.round(distance_matrix_km / speed_knots,decimals=1) # Time in hours





maintenance_times = np.round(generate_maintenance_times(nr_turbines),decimals=2)

header_demand = ['Turbine'] + [f'Type {i+1} Technicians' for i in range(type_technichans)]

# Update turbines_with_base to include charging stations for distance calculations
turbines_with_base_and_chargers = turbines + charging_stations + [base_node]

maintenance_revenues_with_tw = np.round(generate_maintenance_revenues_with_time_windows(nr_turbines, nr_timewindows))


# Save everything including the time matrix to the same CSV file



# Plotting turbines, the base node, and charging stations
plt.figure(figsize=(10, 10))
plt.plot(*zip(*coordinates, coordinates[0]), marker='o', color='red')
plt.scatter([t[0] for t in turbines], [t[1] for t in turbines], color='green', label='Turbines')
plt.scatter([cs[0] for cs in charging_stations], [cs[1] for cs in charging_stations], color='yellow', marker='^', label='Charging Stations')
plt.scatter(base_node[0], base_node[1], color='blue', marker='s', label='Base Node (Haugesund Harbour)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Wind Turbine Layout with Base Node and Charging Stations')
plt.legend()
plt.grid(True)
plt.show()




import csv
import numpy as np

# Assume turbines, charging_stations, maintenance_times, maintenance_revenues, technician_demand are already defined
# Assume distance_matrix_km, cost_matrix_diesel, cost_matrix_electric, battery_usage_matrix, time_matrix_hours are NumPy arrays

with open(f'T{nr_turbines}_C{nr_chargers}all_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    
    # Node locations section
    writer.writerow(['Node_Type', 'Node_ID', 'Longitude', 'Latitude'])
    for i, (lon, lat) in enumerate(turbines):
        writer.writerow(['Turbine', f'Turbine_{i+1}', lon, lat])
    for i, cs in enumerate(charging_stations):
        writer.writerow(['Charging_Station', f'Charging_Station_{i+1}', cs[0], cs[1]])
    writer.writerow(['Base_Node', 'Base_Node', base_node[0], base_node[1]])
    
    # Distance matrix section
    writer.writerow(['DistanceMatrix_NM'])
    header = [''] +['Base_Node'] + [f'Turbine_{i+1}' for i in range(nr_turbines)] + \
            [f'Charging_Station_{i+1}' for i in range(nr_chargers)]
    writer.writerow(header)
    for i, row in enumerate(distance_matrix_km):
        label = header[i+1]  # +1 to skip the empty string at the beginning of the header list
        writer.writerow([label] + list(np.round(row, 2)))

    # Cost matrix for diesel section
    # writer.writerow(['CostMatrix_Diesel_NM'])
    # writer.writerow(header)
    # for i, row in enumerate(cost_matrix_diesel):
    #     label = header[i+1]
    #     writer.writerow([label] + list(np.round(row, 2)))

    # # Cost matrix for electric section
    # writer.writerow(['CostMatrix_Electric_NM'])
    # writer.writerow(header)
    # for i, row in enumerate(cost_matrix_electric):
    #     label = header[i+1]
    #     writer.writerow([label] + list(np.round(row, 2)))

    # # Battery usage matrix section
    # writer.writerow(['BatteryUsageMatrix_Percent'])
    # writer.writerow(header)
    # for i, row in enumerate(battery_usage_matrix):
    #     label = header[i+1]
    #     writer.writerow([label] + list(np.round(row, 2)))

    # # Time matrix section
    # writer.writerow(['TimeMatrix_Hours'])
    # writer.writerow(header)
    # for i, row in enumerate(time_matrix_hours):
    #     label = header[i+1]
    #     writer.writerow([label] + list(np.round(row, 2)))

    # Maintenance tasks section
    writer.writerow(['MaintenanceTasks_Hours'])
    writer.writerow(['Turbine', 'Maintenance_Time_Hours'])
    for i, time in enumerate(maintenance_times):
        writer.writerow([f'Turbine_{i+1}', np.round(time, 2)])

    # Maintenance tasks revenue section
    # writer.writerow(['MaintenanceTasks_Revenue'])
    # writer.writerow(['Turbine', 'Revenue'])
    # for i, revenue in enumerate(maintenance_revenues):
    #     writer.writerow([f'Turbine_{i+1}', np.round(revenue, 2)])
        
    # Write headers for the updated MaintenanceTasks_Revenue section
    writer.writerow(['MaintenanceTasks_Revenue'])
    writer.writerow(['Turbine', 'Revenue', 'TimeWindow'])

# Assuming maintenance_revenues_with_tw contains your flat list of revenue and time window pairs
    for i, (revenue, time_window) in enumerate(maintenance_revenues_with_tw, start=1):
    # Directly calculate the turbine number based on the iteration
    # This assumes each turbine gets exactly 3 entries in maintenance_revenues_with_tw
        t = (i - 1) % nr_turbines + 1
        writer.writerow([f'Turbine_{t}', revenue, time_window])



    # Maintenance task technician demand section
    writer.writerow(['MaintenanceTaskTechnicianDemand'])
    writer.writerow(['Turbine', 'Type_1_Technicians', 'Type_2_Technicians'])
    for i, demand in enumerate(technician_demand):
        writer.writerow([f'Turbine_{i+1}'] + demand)
   
    
    
  



import pandas as pd



data = {
    "Route ID": [],
    "Vessel": [],
    "Time Window": [],
    "Profit": [],
    "Technician Type 1 Demand": [],
    "Technician Type 2 Demand": [],
    "Jobs Included": []
}

# Create a DataFrame
df = pd.DataFrame(data)

#df.to_csv("3t_1c.csv", index=False)
#filnem="variables_per_route_horizontal.csv"
#f = open(filnem, "w+")
#f.close()

# After generating all your parameters and routes, compile them into a single DataFrame
combined_data = {
    "Parameter": ["Vessels", "Time Windows", "Number of Turbines", "Speed Knots", "Price Diesel", "Price Electric", "Max NM Battery", "Type Technicians", "Number of Technicians", "Number of Chargers"],
    "Value": [vessels, nr_timewindows, nr_turbines, speed_knots, price_disel, price_eletric, max_nm_battery, type_technichans, nr_of_technichnas2d, nr_chargers]
    # Add more parameters as necessary
}

# Convert your routes and operational data into a format suitable for CSV output
# This is a simplification; adjust as necessary to include all relevant data
routes_df = pd.DataFrame(data)  # Assuming 'data' holds your dummy routes information

# Combine all data into a single DataFrame (or multiple DataFrames, if more suitable)
# For simplicity, this example just handles parameters; extend as needed
parameters_df = pd.DataFrame(combined_data)

# Output to CSV
parameters_df.to_csv("model_parameters.csv", index=False)
for i in range (1,fleets+1):
    routes_df.to_csv(f'T{nr_turbines}_C{nr_chargers}all_data.csv_fleet{i}.csv',index=False)
