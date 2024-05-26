import ast
import time
import numpy as np
import pandas as pd
from gurobipy import Model, GRB, quicksum
import os

data_file =""

data_files= ["T5_C2all_data.csv","T2_C1all_data.csv"]
route_File = ""







def load_data_sub():
    file_path=data_file

    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]  # This also removes empty lines

    # Helper function to process sections
    def process_section(lines):
        # Assuming the first line contains headers
        headers = lines[0].split(',')
        # Process data rows; splitting each line by ',' and stripping extra whitespace
        data = [line.split(',') for line in lines[1:]]
        
        # Creating DataFrame; ensuring strings are stripped of whitespace
        df = pd.DataFrame(data, columns=[header.strip() for header in headers])
        
        # Reset index to ensure it starts from 0 for each section
        df.reset_index(drop=True, inplace=True)
        
        return df

    # Dictionary to hold all DataFrames
    dfs = {}

    # Temporary storage for current section lines
    current_section_lines = []
    current_section_name = ""

    for line in lines:
        # Check if we hit a new section
        if line in ['Node_Type,Node_ID,Longitude,Latitude', 'DistanceMatrix_NM', 'CostMatrix_Diesel_NM', 'CostMatrix_Electric_NM', 'BatteryUsageMatrix_Percent', 'TimeMatrix_Hours', 'MaintenanceTasks_Hours', 'MaintenanceTasks_Revenue', 'MaintenanceTaskTechnicianDemand']:
            # If there's a current section being processed, save it before moving on
            if current_section_lines:
                dfs[current_section_name] = process_section(current_section_lines)
                current_section_lines = []  # Reset for next section
            current_section_name = line  # Update current section name
        else:
            # Otherwise, we're still collecting lines for the current section
            current_section_lines.append(line)

    # Don't forget to save the last section after the loop ends
    if current_section_lines:
        dfs[current_section_name] = process_section(current_section_lines)
        
    return dfs


def process_jobs(jobs):
    # Ensure that the input is a string
    jobs = str(jobs).strip()
    if not jobs:
        return []
    # Split the jobs by comma, strip whitespace, and convert to integers
    try:
        return [int(job.strip()) for job in jobs.split(',') if job.strip().isdigit()]
    except ValueError:
        # In case of any conversion error, return an empty list or handle as needed
        return []


def load_data():
    df_routes = pd.read_csv(route_File)
    df_parameters = pd.read_csv("model_parameters.csv")
    return df_routes, df_parameters


def initialize_globals():
    global dfs, distanceMatrix, cost_matrix_diesel_df, cost_matrix_electric_df
    global battery_usage_matrix_df, time_matrix_hours_df, maintenance_tasks_hours_df
    global maintenance_tasks_revenue_df, maintenance_task_technician_demand_df
    global df_routes, num_technician_types, df_parameters, pris_disel, pris_el, T
    global total_technicians_availability

    dfs = load_data_sub()
    distanceMatrix = dfs['DistanceMatrix_NM']
    cost_matrix_diesel_df = dfs['CostMatrix_Diesel_NM'] #DC^T
    cost_matrix_electric_df = dfs['CostMatrix_Electric_NM'] #EC^T
    battery_usage_matrix_df = dfs['BatteryUsageMatrix_Percent']
    time_matrix_hours_df = dfs['TimeMatrix_Hours']
    maintenance_tasks_hours_df = dfs['MaintenanceTasks_Hours']
    maintenance_tasks_revenue_df = dfs['MaintenanceTasks_Revenue']
    maintenance_task_technician_demand_df = dfs['MaintenanceTaskTechnicianDemand']
    
    df_routes = pd.read_csv(route_File)
    num_technician_types = df_routes.filter(like='Technician Type').shape[1]
    df_parameters = pd.read_csv("model_parameters.csv")
    
    pris_disel = int(df_parameters.loc[df_parameters["Parameter"] == "Price Diesel", "Value"].values[0])
    pris_el = float(df_parameters.loc[df_parameters["Parameter"] == "Price Electric", "Value"].values[0])
    T = int(df_parameters.loc[df_parameters["Parameter"] == "Time Windows", "Value"].values[0])
    
    total_technicians_availability_str = df_parameters.loc[df_parameters['Parameter'] == 'Number of Technicians', 'Value'].iloc[0]
    total_technicians_availability_list = ast.literal_eval(total_technicians_availability_str)
    total_technicians_availability = {i + 1: (t[0], t[1]) for i, t in enumerate(total_technicians_availability_list)}

# Call the function to initialize the global variables






def build_master_problem(df_routes, df_parameters):
        
    df_routes = pd.read_csv(route_File)
    df_routes['Jobs Included'] = df_routes['Jobs Included'].apply(process_jobs)

    #df_routes['Jobs Included'] = df_routes['Jobs Included'].apply(lambda x: [int(job.strip()) for job in x.split(',')])
    df_parameters = pd.read_csv("model_parameters.csv")
    profits_dict = dict(zip(df_routes['Route ID'], df_routes['Profit']))
    technician_demands = [dict(zip(df_routes['Route ID'], df_routes[f'Technician Type {i} Demand'])) for i in range(1, num_technician_types + 1)]
    vessels = df_routes['Vessel'].unique()   #maybe take from parameters file to get number??#curently not used wil clean
    time_windows = df_routes['Time Window'].unique()
    #print(total_technicians_availability)
            
    # Initialize the model
    model = Model("Master Problem")

    # creat binary varibal x in the model 
    route_vars = model.addVars(df_routes['Route ID'], vtype=GRB.CONTINUOUS, name="route", lb=0)


    # Set the objective function
    model.setObjective(sum(profits_dict[r] * route_vars[r] for r in df_routes['Route ID']), GRB.MAXIMIZE)

    for t in range(1, T + 1):
    # Access the total number of technicians available for this time window
        total_tech_for_time_window = total_technicians_availability[t]

        # Filter routes based on the current time window t
        routes_in_current_tw = df_routes[df_routes['Time Window'] == t]['Route ID']

        for i in range(1, num_technician_types + 1):
            # Calculate the total demand for technicians of type i in time window t,
            # considering only routes operating in this time window
            total_demand_for_tech_type = sum(route_vars[r] * technician_demands[i-1].get(r, 0) for r in routes_in_current_tw)

            # Add the constraint for this technician type and time window
            model.addConstr(total_demand_for_tech_type <= total_tech_for_time_window[i-1], f"tech_type_{i}_time_window_{t}_availability")


    job_route_mapping = {}
    for idx, row in df_routes.iterrows():
        # Assuming "Jobs Included" is already a list of integers
        jobs = row['Jobs Included']
        for job in jobs:
            job = str(job)  # Convert job number to string if needed for consistent key usage
            if job not in job_route_mapping:
                job_route_mapping[job] = []
            job_route_mapping[job].append(row['Route ID'])

    # Constraint: Ensure each job is done no more than once across all routes
    for job, routes in job_route_mapping.items():
        model.addConstr(sum(route_vars[route] for route in routes) <= 1, f"job_{job}_once")


    # Iterate over each vessel and time window to add the constraint
    #constraint 4.5 
    for vessel in df_routes['Vessel'].unique():
        for time_window in range(1,T+1) :
            # Filter the DataFrame for the current vessel and time window
            relevant_routes = df_routes[(df_routes['Vessel'] == vessel) & (df_routes['Time Window'] == time_window)]
            
            # Sum over routes for the current vessel and time window and ensure the sum does not exceed 1
            model.addConstr(sum(route_vars[route] for route in relevant_routes['Route ID']) <= 1, 
                            name=f"one_route_per_vessel_{vessel}_tw_{time_window}")

    return model, route_vars



def build_master_problem_IP(df_routes, df_parameters):
        
    df_routes = pd.read_csv(route_File)
    df_routes['Jobs Included'] = df_routes['Jobs Included'].apply(process_jobs)

    #df_routes['Jobs Included'] = df_routes['Jobs Included'].apply(lambda x: [int(job.strip()) for job in x.split(',')])
    df_parameters = pd.read_csv("model_parameters.csv")
    profits_dict = dict(zip(df_routes['Route ID'], df_routes['Profit']))
    technician_demands = [dict(zip(df_routes['Route ID'], df_routes[f'Technician Type {i} Demand'])) for i in range(1, num_technician_types + 1)]
    vessels = df_routes['Vessel'].unique()   #maybe take from parameters file to get number??#curently not used wil clean
    time_windows = df_routes['Time Window'].unique()
    #print(total_technicians_availability)
            
    # Initialize the model
    model_IP = Model("Master Problem IP")

    # creat binary varibal x in the model 
    route_vars = model_IP.addVars(df_routes['Route ID'], vtype=GRB.BINARY, name="route")


    # Set the objective function
    model_IP.setObjective(sum(profits_dict[r] * route_vars[r] for r in df_routes['Route ID']), GRB.MAXIMIZE)

    for t in range(1, T + 1):
    # Access the total number of technicians available for this time window
        total_tech_for_time_window = total_technicians_availability[t]

        # Filter routes based on the current time window t
        routes_in_current_tw = df_routes[df_routes['Time Window'] == t]['Route ID']

        for i in range(1, num_technician_types + 1):
            # Calculate the total demand for technicians of type i in time window t,
            # considering only routes operating in this time window
            total_demand_for_tech_type = sum(route_vars[r] * technician_demands[i-1].get(r, 0) for r in routes_in_current_tw)

            # Add the constraint for this technician type and time window
            model_IP.addConstr(total_demand_for_tech_type <= total_tech_for_time_window[i-1], f"tech_type_{i}_time_window_{t}_availability")


    job_route_mapping = {}
    for idx, row in df_routes.iterrows():
        # Assuming "Jobs Included" is already a list of integers
        jobs = row['Jobs Included']
        for job in jobs:
            job = str(job)  # Convert job number to string if needed for consistent key usage
            if job not in job_route_mapping:
                job_route_mapping[job] = []
            job_route_mapping[job].append(row['Route ID'])

    # Constraint: Ensure each job is done no more than once across all routes
    for job, routes in job_route_mapping.items():
        model_IP.addConstr(sum(route_vars[route] for route in routes) <= 1, f"job_{job}_once")


    # Iterate over each vessel and time window to add the constraint
    #constraint 4.5 
    for vessel in df_routes['Vessel'].unique():
        for time_window in range(1,T+1) :
            # Filter the DataFrame for the current vessel and time window
            relevant_routes = df_routes[(df_routes['Vessel'] == vessel) & (df_routes['Time Window'] == time_window)]
            
            # Sum over routes for the current vessel and time window and ensure the sum does not exceed 1
            model_IP.addConstr(sum(route_vars[route] for route in relevant_routes['Route ID']) <= 1, 
                            name=f"one_route_per_vessel_{vessel}_tw_{time_window}")

    return model_IP, route_vars


##maybe routvars is only for first iteration check later

def optimize_master_problem(model, df_routes, route_vars):
    solution = {}
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print("Optimal solution found:")
        print(f"Total Profit: {model.ObjVal}")
        model.write("Masterproblem.lp")  # Write out the model to an LP file
    
        for r in df_routes['Route ID']:
            if route_vars[r].X > 0.01:
                solution[r] = route_vars[r].X
        solution['Total Profit'] = model.ObjVal
    else:
        print("Optimal solution not found or model did not solve to optimality.")
    return solution

def extract_dual_prices_1(model, df_routes, num_technician_types, V, T):
    dual_prices = {}

    # Technician availability dual prices (Not dependent on V)
    for t in range(1, T + 1):
        for i in range(1, num_technician_types + 1):
            constraint_name = f"tech_type_{i}_time_window_{t}_availability"
            constraint = model.getConstrByName(constraint_name)
            if constraint:
                # Directly storing without looping over vessels, as these constraints are not vessel-specific
                dual_prices[constraint_name] = constraint.Pi

    job_route_mapping = {}
    for idx, row in df_routes.iterrows():
        # Properly process the 'Jobs Included' column for each route
        jobs = process_jobs(row['Jobs Included'])
        for job in jobs:
            job = str(job).strip()  # Ensure the job is a string and stripped
            if job not in job_route_mapping:
                job_route_mapping[job] = []
            job_route_mapping[job].append(row['Route ID'])

    for job, routes in job_route_mapping.items():
        constraint_name = f"job_{job}_once"
        constraint = model.getConstrByName(constraint_name)
        if constraint:
            dual_prices[constraint_name] = constraint.Pi

    # Vessel-time window dual prices (Specifically dependent on V and T)
    for vessel in df_routes['Vessel'].unique():
        for time_window in df_routes['Time Window'].unique():
            constraint_name = f"one_route_per_vessel_{vessel}_tw_{time_window}"
            constraint = model.getConstrByName(constraint_name)
            if constraint:
                # Including vessel and time window in the key as these constraints are specific to each vessel and time window
                dual_prices[(constraint_name)] = constraint.Pi

    # Preparing data for CSV output, including handling cases where there's no direct vessel or time window association
    dual_prices_list = []
    for key, value in dual_prices.items():
        if isinstance(key, tuple):
            name, t, v = key
        else:
            name = key
            t, v = "-", "-"  # Indicating non-applicability for technician and job constraints

        dual_prices_list.append({"Dual Name": name, "Dual Value": value})

    df_dual_prices = pd.DataFrame(dual_prices_list)
    df_dual_prices.to_csv('corrected_dual_prices.csv', index=False)


    return dual_prices


def define_nodes(df_parameters):
    m = int(df_parameters[df_parameters['Parameter'] == 'Number of Turbines']['Value'].values[0])
    k = int(df_parameters[df_parameters['Parameter'] == 'Number of Chargers']['Value'].values[0])
    n = k + m
    nodes = {
        'delivery': list(range(1, n+1)),
        'pickup': list(range(n+1, 2*n+1)),
        'charging_delivery': list(range(m+1, m+k+1)),
        'charging_pickup': list(range(n+m+1, n+m+k+1)),
        'origin_destination': [0, 2*n+1]
    }

    return nodes, n


def construct_arcs(nodes, n):

    arcs = {}

    # Helper function to add an arc with attributes
    def add_arc(i, j, attributes):
        arcs[(i, j)] = attributes

    # From the origin node to all delivery nodes
    for j in nodes['delivery']:
        add_arc(0, j, {'type': 'origin_to_delivery'})

    # From delivery nodes to other delivery nodes, excluding charger delivery nodes
    for i in nodes['delivery']:
        if i not in nodes['charging_delivery']:  # Exclude charger delivery nodes
            for j in nodes['delivery']:
                if i != j:  # Avoid loop to the same node
                    add_arc(i, j, {'type': 'delivery_to_delivery'})

    # From delivery nodes to pick-up nodes, excluding charger pick-up nodes
    for i in nodes['delivery']:
        if i not in nodes['charging_delivery']:  # Exclude charger delivery nodes
            for j in nodes['pickup']:
                if j not in nodes['charging_pickup']:  # Exclude charger pick-up nodes
                    add_arc(i, j, {'type': 'delivery_to_pickup'})

    # From charger delivery nodes to corresponding pick-up nodes
    for i in nodes['charging_delivery']:
        corresponding_pickup = i + n  # Adjusted mapping for simplicity
        add_arc(i, corresponding_pickup, {'type': 'charging'})

    # From pick-up nodes to the destination node
    for i in nodes['pickup']:
        add_arc(i, 2*n + 1, {'type': 'pickup_to_destination'})

    # From pick-up nodes to all delivery nodes, excluding the corresponding delivery node
    for i in nodes['pickup']:
        for j in nodes['delivery']:
            if j != i - n:  # Exclude the corresponding delivery node
                add_arc(i, j, {'type': 'pickup_to_delivery'})

    # From pick-up nodes to other pick-up nodes, excluding self
    for i in nodes['pickup']:
        for j in nodes['pickup']:
            if i != j:  # Exclude self
                add_arc(i, j, {'type': 'pickup_to_pickup'})

    return arcs



############SUBPROBLEM#############################################################


# Initialize F_bi dictionary
def generate_technician_demand(demand_df, num_technician_types, n):

    F_bi = {}
    
    # Iterate over the rows of the technician demand DataFrame
    for idx, row in demand_df.iterrows():
        # Assume Node_ID in 'Turbine_X' format and we want just the number X
        node_id = int(row['Turbine'].split('_')[1])
        
        # For each technician type, create a key in F_bi with the node_id and the demand
        for tech_type in range(1, num_technician_types + 1):
            tech_demand = row[f'Type_{tech_type}_Technicians']
            if pd.notnull(tech_demand):  # Check if the technician demand is not NaN
                F_bi[(tech_type, node_id)] = int(tech_demand)
                # Also, create the negative demand for the corresponding pickup node
                pickup_node_id = node_id + n
                F_bi[(tech_type, pickup_node_id)] = -int(tech_demand)
    
    return F_bi


def matrix_to_dict(cost_df,n):
    
    cost_of_route = {}
    
    for row in range(0, 1):  # Loop through rows
        for col in range(1, cost_df.shape[1]):  # Start from 1 to exclude first column
            cost_of_route[(row, col-1)] = cost_df.iloc[row, col]

    for row in range(0, 1):  # Loop through rows
        for col in range(1, cost_df.shape[1]-1):  # Start from 1 to exclude first column
            cost_of_route[(row, col+n)] = cost_df.iloc[row, col+1]
    
    for row in range(0, cost_df.shape[0]):  # Loop through rows
        for col in range(1, 2):  # Start from 1 to exclude first column
            cost_of_route[(row, col-1)] = cost_df.iloc[row, col]
    
    for row in range(0, cost_df.shape[0]):  # Loop through rows
        for col in range(1, 2):  # Start from 1 to exclude first column
            cost_of_route[(row+n, col-1)] = cost_df.iloc[row, col]


    for row in range(1, cost_df.shape[0]):  # Loop through rows
        for col in range(2, cost_df.shape[1]):  # Start from 1 to exclude first column
            cost_of_route[(row, col-1)] = cost_df.iloc[row, col]
    
    for row in range(1, cost_df.shape[0]):  # Loop through rows
        for col in range(2, cost_df.shape[1]):  # Start from 1 to exclude first column
            cost_of_route[(row+n, col-1)] = cost_df.iloc[row, col]
            

    for row in range(1, cost_df.shape[0]):  # Loop through rows
        for col in range(2, cost_df.shape[1]):  # Start from 1 to exclude first column
            cost_of_route[(row, col-1+n)] = cost_df.iloc[row, col]
            
    for row in range(1, cost_df.shape[0]):  # Loop through rows
        for col in range(2, cost_df.shape[1]):  # Start from 2 to exclude first column
            cost_of_route[(row+n, col-1+n)] = cost_df.iloc[row, col]
            
    
    for row in range(0, 1):  # Loop through rows
        for col in range(2, cost_df.shape[1]):  # Start from 1 to exclude first column
            cost_of_route[(n+n+1, col-1+n)] = cost_df.iloc[row, col]
            # print(n+n+1, col-1+n)
            
    for row in range(0, 1):  # Loop through rows
            for col in range(2, cost_df.shape[1]):  # Start from 1 to exclude first column
                cost_of_route[(col-1+n, n+n+1)] = cost_df.iloc[row, col]
                # print(col-1+n, n+n+1) 
                
    for row in range(0, cost_df.shape[0]):  # Loop through rows
        for col in range(1, 2):  # Start from 1 to exclude first column
            cost_of_route[(row+n, col-1+n)] = cost_df.iloc[row, col]
           # print(row+n, col-1+n)
            
    for row in range(0, cost_df.shape[0]):  # Loop through rows
        for col in range(1, 2):  # Start from 1 to exclude first column
            cost_of_route[(col-1+n, row+n)] = cost_df.iloc[row, col]
            #print(col-1+n, row+n)

    return cost_of_route


def vessel_attributes(v):
    if Fleet ==1:
      K_v,forbruk,speed,epsilon,DPC= vessel_attributes_fleet1(v)
    elif Fleet == 2:
        K_v,forbruk,speed,epsilon,DPC= vessel_attributes_fleet2(v)
    elif Fleet == 3:
        K_v,forbruk,speed,epsilon,DPC= vessel_attributes_fleet3(v)
    else:
        print("wrong Fleet")
    return K_v ,forbruk,speed,epsilon,DPC
        





def vessel_attributes_fleet1(v):
    #CRC SENTINEL 
    if v == 1:
        K_v = 12
        forbruk = 400 #liter per time
        speed = 21
        epsilon = 0.00001 #ungå delt på null 

        # Convert string to float and multiply by 0.8
    elif v == 2:
        #CRC VULCAN
        K_v = 12
        forbruk = 270 #liter per time
        speed = 21
        epsilon = 0.00001
    else:
        #CRC GLADIATOR
        K_v = 12
        forbruk = 160 #liter per time
        speed = 21
        epsilon = 0.00001
    DPC = 0.6*forbruk * 2 #set DPC as 20percent of service speed consumtion, when no battery
    return K_v,forbruk,speed,epsilon,DPC


def vessel_attributes_fleet2(v):
    #pioner
    if v == 1:
        K_v = 24
        forbruk = 180 #liter per time
        speed = 27
        epsilon =0.00001
        DPC = 0.05 * forbruk     #set dPC as 5 persent of service speed when battery

    #sprinter 
    elif v == 2:

        K_v = 24
        forbruk = 578 #liter per time
        speed = 35
        epsilon =0.0001
        DPC = 0.2*forbruk #set DPC as 20percent of service speed consumtion, when no battery


    else:
        K_v = 0
        forbruk = 10000000 # incase i forget to set correct nr of vessels
        speed =0
        epsilon=0.000001
        DPC = 1*forbruk 

    return K_v,forbruk,speed,epsilon, DPC



def vessel_attributes_fleet3(v):
    #Pioneer2.0
    K_v = 24
    forbruk = 180 #liter per time
    DPC = forbruk * 0.05 #set dPC as 5 persent of service speed when battery
    speed = 27
    epsilon = 60
    # Convert string to float and multiply by 0.8

    return K_v,forbruk,speed,epsilon,DPC




def revenue_to_dict_with_time(revenue_df, t):
    # Convert TimeWindow to float and then to int for accurate comparison
    revenue_df['TimeWindow'] = revenue_df['TimeWindow'].astype(float).astype(int)

    # Filter the DataFrame for rows where the TimeWindow matches the specified t (as an integer)
    filtered_df = revenue_df[revenue_df['TimeWindow'] == t]

    # Use a dictionary comprehension to iterate over each row in the filtered DataFrame
    revenue_dict = {
        row['Turbine']: {'Revenue': float(row['Revenue']), 'TimeWindow': row['TimeWindow']}
        for _, row in filtered_df.iterrows()
    }

    return revenue_dict


def parse_dual_prices(dual_prices, t, v, csv_path="dual.csv"):

    lambda_values = {k: i for k, i in dual_prices.items() if k.startswith(f"one_route_per_vessel_Vessel_{v}_tw_{t}")}
    omega_values = {k: i for k, i in dual_prices.items() if k.startswith(f"tech_type_") and f"time_window_{t}" in k}
    rho_values = {k: i for k, i in dual_prices.items() if k.startswith("job_")}
    
    # Combine all dual prices into one dictionary for easier processing
    all_dual_prices = {**lambda_values, **omega_values, **rho_values}
    
    # Prepare data for DataFrame
    data = {
        'Dual Name': list(all_dual_prices.keys()),
        'Time Window (t)': [t] * len(all_dual_prices),
        'Vessel (v)': [v] * len(all_dual_prices),
        'Dual Value': list(all_dual_prices.values())
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Check if the CSV file exists and is not empty
    if os.path.isfile(csv_path):
        try:
            # Attempt to read the file to determine if it's truly empty
            pd.read_csv(csv_path)
            # If no error, the file exists and has content; no header needed
            header = False
        except pd.errors.EmptyDataError:
            # If an error occurs, the file is empty; header needed
            header = True
    else:
        # If the file doesn't exist, we'll need to write a header
        header = True
    
    # Append to CSV, create if doesn't exist, else append without headers if the file already has content
    df.to_csv(csv_path, mode='a', header=header, index=False)
    
    return lambda_values, omega_values, rho_values

import pandas as pd

def parse_dual_prices1(dual_prices, t, v):
    # Initialize dictionaries to hold filtered dual prices
    lambda_values = {}
    omega_values = {}
    rho_values = {}

    # Process the dual prices
    for key, value in dual_prices.items():
        if key.startswith("one_route_per_vessel_Vessel_"):
            # Extract vessel number and time window from the key
            parts = key.split("_")
            vessel_number = int(parts[4])  # Assuming vessel number is the 5th element
            time_window = int(parts[6])  # Assuming time window is the 7th element
            
            if vessel_number == v and time_window == t:
                lambda_values[key] = value
        elif key.startswith("tech_type_"):
            # Assuming tech_type constraints include the specific time window in their name
            if f"time_window_{t}" in key:
                 omega_values[key] = value
        elif key.startswith("job_"):
            rho_values[key] = value

    return lambda_values, omega_values, rho_values


def build_sub_problem(nodes, arcs, dual_prices, n, t, v, df_routes, num_technician_types, max_route_id,runtime):
    #print(arcs)

    ######midlertidig######33
    TP=16

    #timer når den må komme til destinasjon
    ## wil get from model paramters.csv later
    L = 100 # charges 100 nm per hour, means aprox 2666 kw, C-rate of 0.6?
    kappa = 0.5
    pi = 0.1
    bat_kwh =1618

    V = df_routes['Vessel'].unique()
    subproblem = Model("Subproblem")
    subproblem.setParam('TimeLimit', runtime)
    subproblem.setParam('MIPGap', 0.1)

    y_ij = subproblem.addVars(arcs.keys(), vtype=GRB.BINARY, name="y")
    # Add continuous variables for electric and diesel usage proportions
    E_ij = subproblem.addVars(arcs.keys(), vtype=GRB.CONTINUOUS, name="E", lb=0, ub=1)
    D_ij = subproblem.addVars(arcs.keys(), vtype=GRB.CONTINUOUS, name="D", lb=0, ub=1)
    # Parse dual prices from the master problem
    lambda_values, omega_values, rho_values = parse_dual_prices(dual_prices, t, v)
    K_v ,forbruk,speed,epsilon,DPC = vessel_attributes(v)

    # Convert the revenue and cost data frames into dictionaries

    distance = matrix_to_dict(distanceMatrix,n)
    Tij  ={key: str(float(value) / speed) for key, value in distance.items()}

    

    usage = matrix_to_dict(distanceMatrix,n)
    
    # diesel_costs = matrix_to_dict(dfs['CostMatrix_Diesel_NM'],n)


    # #print(diesel_costs)
    # electric_costs = matrix_to_dict(dfs['CostMatrix_Electric_NM'],n)


    disel =forbruk * pris_disel 
    diesel_costs ={key: str(float(value) * disel) for key, value in Tij.items()} 
    eletric =(bat_kwh/epsilon) * pris_el  ## kan byttes me variabel maxbattery_kwh, maxbatteri i kwh / max batteri i nm gir hvor mange kw for å resise 1 nm,, kwh per nm * pris (kr/kwh) pris for å reise i nm, neste ganger med nm reist
    electric_costs ={key: str(float(value) * eletric) for key, value in distance.items()}#blir avstand (nm) * maxbatteri i kwh / max batteri i nm  * antall nm * pris/kwh som gir 

    #print (electric_costs)
    
    
    F_bi = generate_technician_demand(dfs['MaintenanceTaskTechnicianDemand'], num_technician_types, n)
    Ti = maintenance_tasks_hours_df.set_index('Turbine')['Maintenance_Time_Hours'].to_dict()
    
    revenues = revenue_to_dict_with_time(dfs['MaintenanceTasks_Revenue'],t)
    G_bt = total_technicians_availability[t]
    ND = nodes['delivery']
    NP = nodes['pickup']
    NDC = nodes['charging_delivery']  # Delivery nodes with charging stations
    NPC = nodes['charging_pickup']  # Pickup nodes with charging stations
    NC = NDC+ NPC
    NDnotC =[]
    for i in ND:
        if i not in NDC:
            NDnotC.append(i)


    A = arcs
    OD = nodes['origin_destination']
    N = list(range(2*n+2))  # Includes all nodes from 0 to 2*n+1
    theta_i = subproblem.addVars(N, vtype=GRB.CONTINUOUS, name="theta", lb=0, ub=epsilon)  # Assuming 100 is the battery capacity
    z_bi = subproblem.addVars([(b, i) for b in range(1, num_technician_types + 1) for i in N], vtype=GRB.INTEGER, name="z", lb=0)
    Tdp = subproblem.addVars(NDnotC,vtype=GRB.CONTINUOUS,name="Tdp",lb=0)
    q = subproblem.addVars(N, vtype=GRB.CONTINUOUS, name="q")  # Time vessel leaves node i

#####################travelconstraints###################################################################
    
    subproblem.addConstr(quicksum(y_ij[(0, j)] for j in ND) == 1, "leave_origin_once")
    # Correcting the return_destination_once constraint
    subproblem.addConstr(quicksum(y_ij[(i, 2*n+1)] for i in NP) == 1, "return_destination_once")

      # Balance constraint: Vessel leaves each node as many times as it arrives
    
    for i in ND + NP:  # dont include N^od
        subproblem.addConstr(
            quicksum(y_ij[j, i] for j in N if (j, i) in arcs) -
            quicksum(y_ij[i, j] for j in N if (i, j) in arcs) == 0,
            f"flow_conservation_node_{i}")
    for j in ND:
        # Ensure each delivery node j is linked to its corresponding pickup node j+n
        subproblem.addConstr(
            quicksum(y_ij[i, j] for i, _j in arcs if _j == j) - 
            quicksum(y_ij[i, j+n] for i, _j in arcs if _j == j+n) == 0, 
            f"pickup_after_delivery_{j}"
    )

# ###########men ikke med i modelen egt #####################
    for j in N:  #
        if j not in NC:  # Exclude charger nodes from this constraint
            subproblem.addConstr(quicksum(y_ij[i, j] for i in N if (i, j) in arcs) <= 1, f"visit_turbine_{j}_once")

##########################eletricconstraint###############################################
       # Battery and Fuel Constraints
    subproblem.addConstr(theta_i[0] == epsilon, "initial_battery_charge")

    for i in N:
        subproblem.addConstr(theta_i[i] <= epsilon, f"max_battery_capacity_{i}")
        #subproblem.addConstr(theta_i[i] >= 0.1, f"check{i}")

    for (i, j) in arcs:
        subproblem.addConstr(E_ij[i, j] + D_ij[i, j] == y_ij[i, j], f"energy_mix_{i}_{j}")

    for i, j in A:
        if i not in NDC:
            batcost = float(usage.get((i, j), 1))
            #print(batcost)
            # Constraint skips charger nodes
            subproblem.addConstr(theta_i[j] <= theta_i[i] - batcost * E_ij[i, j] + epsilon * (1 - y_ij[i, j]),
                             name=f"batteryUpdate{i}_{j}")
    for i in NDC: # 
        subproblem.addConstr(
            theta_i[i] + L * (q[i + n] - q[i]-kappa) >= theta_i[i + n],
            f"charging_at_node_{i}"
        )
    # subproblem.addConstr(quicksum(E_ij[(0, j)] for j in ND) >=pi, "leave_origin_on_eletric")

    # subproblem.addConstr(quicksum(E_ij[(i, 2*n+1)] for i in NP) >=pi, "enter_base_on_eletric")



##############time constraints#################################################################
        
    

    subproblem.addConstr(q[(2*n+1)] <= TP, name="return_to_destination")


    for i, j in A:
        travel_time = float(Tij.get((i, j), 0))
        #print(float(Tij.get((i, j), 0)))  # Convert the travel time to float and provide a default of 0
        subproblem.addConstr(q[i] + travel_time <= (TP + travel_time) * (1 - y_ij[i, j]) + q[j],
                            name=f"time_update_{i}_{j}")

    for i in N:
        turbine_key = f'Turbine_{i}'
        if turbine_key in Ti:
            # Convert the maintenance time from string to float
            maintenance_time = float(Ti[turbine_key])
            subproblem.addConstr(q[i] + maintenance_time <= q[i + n], name=f"adequate_time_for_tasks_{i}")

    ################dynamicpositioning####################
    for i in NDnotC:
        subproblem.addConstr((q[i + n] - q[i]) + (y_ij[i,i+n]-1)*TP<= Tdp[i])
   
################technichans##################################################################
    subproblem.addConstr(quicksum(z_bi[(b, 0)] for b in range(1, num_technician_types+1 )) <= K_v, "vessel_tech_capacity")
    


    for b in range(1, num_technician_types + 1):
        subproblem.addConstr(z_bi[(b,0)]<= G_bt[b-1],"tech_availibility")

    for b in range(1, num_technician_types+1 ):
        for i, j in A:  # For each arc in the arc set
            subproblem.addConstr(
                z_bi[b, i] - F_bi.get((b, j), 0) <= z_bi[b, j] + K_v * (1 - y_ij[i, j]),
                name=f"tech_availability_meet_demand_{b}_{i}_{j}"
            )
            subproblem.addConstr(
                z_bi[b, i] - F_bi.get((b, j), 0) >= z_bi[b, j] - K_v * (1 - y_ij[i, j]),
                name=f"tech_availability_no_overestimate_{b}_{i}_{j}"
            )


################################objF################################################################

    subproblem.update()   
    
  

    objective = quicksum(
        y_ij[(i, j)] * ((float(revenues[f"Turbine_{j}"]['Revenue']) - rho_values.get(f"job_{j}_once", 0)))
        for i, j in arcs if f"Turbine_{j}" in revenues
    ) - quicksum(
        E_ij[(i, j)] * float(electric_costs.get((i, j), 1)) 
         + D_ij[(i, j)] * float(diesel_costs.get((i, j), 1))
        for i, j in arcs.keys()

    ) - quicksum(
        Tdp[(i)]* DPC
        for i in ND if i not in NDC


    ) - quicksum(
        z_bi[(b, 0)] * omega_values.get(f"tech_type_{b}_time_window_{t}_availability",0)
        for b in range(1, num_technician_types + 1)

    )-lambda_values.get(f"one_route_per_vessel_Vessel_{v}_tw_{t}", 0)

    subproblem.setObjective(objective, GRB.MAXIMIZE)

    # Set the objective in the model
    print('Objective function components:')
    print(f"for time window {t} and vessel {v}")
    subproblem.update()  
    subproblem.write("subproblem.lp")  # Write the model to an LP file
    

    #print('Objective:', subproblem.getObjective())
    

    print('Number of constraints:', subproblem.NumConstrs)
    
    #subproblem.optimize()
    
    subproblem.write("subproblem.lp")
    
    # Now, optimize the model
    subproblem.optimize()
    
    # Check if the solution is optimal or,  it has a positive objective value, terminated due to time limit
    if ((subproblem.Status == GRB.OPTIMAL and subproblem.ObjVal > 2)):
        objective_value = subproblem.ObjVal
        new_routes, updated_max_route_id = extract_new_routes_from_subproblem(subproblem, arcs, ND, NC, n, t, v, df_routes, num_technician_types, max_route_id,electric_costs,diesel_costs,revenues,DPC)
        
        return subproblem, new_routes, updated_max_route_id,objective_value,True
    


    elif (subproblem.Status == GRB.TIME_LIMIT and subproblem.ObjVal > 2):
        objective_value = subproblem.ObjVal
        
        new_routes, updated_max_route_id = extract_new_routes_from_subproblem(subproblem, arcs, ND, NC, n, t, v, df_routes, num_technician_types, max_route_id,electric_costs,diesel_costs,revenues,DPC)
        return subproblem, new_routes, updated_max_route_id,objective_value,False


        
    else:
        return subproblem, [], max_route_id, -1,True


def write_variables_to_csv_horizontal(subproblem, route_id, csv_filename="variables_per_route_horizontal.csv"):
    # Create a dictionary for routes
    data_for_route = {"Route ID": route_id+1}
    
    # Iterate through variables in the subproblem
    for var in subproblem.getVars():
        if var.X > 0:  
            data_for_route[var.VarName] = (f'{var.varName}: {var.x}')
    
    # Convert the dictionary to a DataFrame, but first ensure it is in a list to create one row
    df_data = pd.DataFrame([data_for_route])
    
    # Determine whether to write a header
    header = not os.path.exists(csv_filename)  # Write header only if the file does not exist
    
    # Write or append the DataFrame to a CSV file
    df_data.to_csv(csv_filename, mode='a', header=header, index=False)







def extract_new_routes_from_subproblem(subproblem, arcs, ND, NC, n, t, v, df_routes, num_technician_types, max_route_id, electric_costs, diesel_costs, revenues,DPC):
    max_route_id += 1
    technician_demands = {b: int(subproblem.getVarByName(f"z[{b},0]").X) for b in range(1, num_technician_types + 1)}
    obj = subproblem.ObjVal
    # Initialize containers for the variables
    y_ij = {}
    E_ij = {}
    D_ij = {}
    TDP={}
    jobs = set()




    # Retrieve and process all variables from the subproblem model
    for var in subproblem.getVars():
        var_name = var.VarName
        if var_name.startswith("y["):
            i, j = map(int, var_name[2:-1].split(','))
            y_ij[(i, j)] = var.X
        elif var_name.startswith("E["):
            i, j = map(int, var_name[2:-1].split(','))
            E_ij[(i, j)] = var.X
        elif var_name.startswith("D["):
            i, j = map(int, var_name[2:-1].split(','))
            D_ij[(i, j)] = var.X
        elif var_name.startswith("Tdp["):
            i = int(var_name[4:-1])  # Extracting the index and converting to an integer
            TDP[i] = var.X  # Storing the value in the dictionary with key `i`



    # Initialize profit calculation
    rev = 0
    cost = 0

    # Summing revenues for each job included in the route
    for i, j in arcs:
        if  y_ij[(i, j)] >0.1 and f"Turbine_{j}" in revenues:
            rev += float(revenues[f"Turbine_{j}"]['Revenue'])
            jobs.add(j)


        

    # Calculating costs associated with each arc traversed, both electric and diesel
    for i, j in arcs:
        if (i, j) in y_ij and y_ij[(i, j)] == 1:
            cost += E_ij[(i, j)] * float(electric_costs.get((i, j), 1))
            cost += D_ij[(i, j)] * float(diesel_costs.get((i, j), 1))
    cost_dp = 0
    for i in ND:
        if i not in NC:

            cost_dp+=TDP.get(i) * float(DPC)
            cost += TDP.get(i) * float(DPC)



    

    profit = rev - cost

    # Prepare the new route information
    # Prepare the new route information
    new_route = {
    "Route ID": max_route_id,
    "Vessel": f"Vessel_{v}",
    "Time Window": t,
    "Profit": profit,
    "Technician Type 1 Demand": technician_demands.get(1, 0),
    "Technician Type 2 Demand": technician_demands.get(2, 0),
    "Jobs Included": ','.join(map(str, sorted(jobs))),
      "obj price": obj,
      "cost dp": cost_dp
          # Updated to remove extra quotes
}


    new_routes = [new_route]
    write_variables_to_csv_horizontal(subproblem, max_route_id-1)
    return new_routes, max_route_id
    # if profit>2:
    #     return new_routes, max_route_id
    # else:
    #     return [], max_route_id-1


def update_routes_df_and_csv(new_routes, df_routes, csv_path=route_File):
    # Convert new routes to Data1Frame
    df_new_routes = pd.DataFrame(new_routes)
    
    # Append new routes to the existing DataFrame
    df_routes_updated = pd.concat([df_routes, df_new_routes], ignore_index=True)
    
    # Save updated DataFrame to CSV
    df_routes_updated.to_csv(csv_path, index=False)
    return df_routes_updated

def checkwotime( nodes, arcs, dual_prices, n, t, v, df_routes, num_technician_types, max_route_id,time):

    for v in range(1, V + 1):  # Assuming V is defined
        for t in range(1, T + 1):  # Assuming T is defined
            # Solve the subproblem for each vessel and time window
            subproblem, new_routes, updated_max_route_id,obj = build_sub_problem(
                nodes, arcs, dual_prices, n, t, v, df_routes, num_technician_types, max_route_id,time)
            if obj >1:
                df_routes = update_routes_df_and_csv(new_routes, df_routes, route_File)
                max_route_id = updated_max_route_id 
                return True
            
    return False
    

def append_solution_to_csv(df_routes, lp_solution, ip_solution, csv_file, duration):
    # Open the CSV file in append mode
    with open(csv_file, 'a') as file:
        # Write the LP solution
        if 'Total Profit' in lp_solution:
            lp_routes = ", ".join([f"{route}:{lp_solution[route]}" for route in lp_solution if route != 'Total Profit'])
            file.write(f"\nTotal Profit (LP),,,,,,,,,,,{lp_solution['Total Profit']} Routes in optimal [{lp_routes}]\n")
        
        # Write the IP solution
        if 'Total Profit' in ip_solution:
            ip_routes = ", ".join([f"{route}:{ip_solution[route]}" for route in ip_solution if route != 'Total Profit'])
            file.write(f"\nTotal Profit (IP),,,,,,,,,,,{ip_solution['Total Profit']} Routes in optimal [{ip_routes}]\n")

        # Write the duration
        file.write(f"\nExecution Time (seconds),,,,,,,,,,,{duration}\n")


def add_total_profit_to_df(solution, df_routes, solution_type):
    for entry in solution:
        if entry.startswith('Total Profit'):
            total_profit = float(entry.split(":")[1])
            new_row = {
                'Route ID': f'Total Profit ({solution_type})',
                'Total Profit': total_profit
            }
            df_routes = df_routes.append(new_row, ignore_index=True)
    
    return df_routes




def run_column_generation(timelimit1,timelimit2,route_File):
    start_time = time.time() 
    # Load initial data and build the master problem
    df_routes, df_parameters = load_data()
    nodes, n = define_nodes(df_parameters)
    arcs = construct_arcs(nodes, n)
    # Initialize max_route_id based on the current maximum route ID in df_routes
    if not df_routes.empty:
        max_route_id = df_routes['Route ID'].max()
    else:
        max_route_id = 0

    improvement_found = True  # Flag to indicate if an improvement was found in the iteration

    while improvement_found:
        # Assume no improvement will be found in this iteration
        improvement_found = False
        improve_count = 0
        optimality = True
        # Solve the master problem and get dual values
        model, route_vars = build_master_problem(df_routes, df_parameters)
        s = optimize_master_problem(model, df_routes, route_vars)
        dual_prices = extract_dual_prices_1(model, df_routes, num_technician_types,V,T)

        #Loop over each vessel and time window
        for v in range(1, V + 1):  # Assuming V is defined
            for t in range(1, T + 1):  # Assuming T is defined
                # Solve the subproblem for each vessel and time window
                subproblem, new_routes, updated_max_route_id,obj,optimal = build_sub_problem(
                    nodes, arcs, dual_prices, n, t, v, df_routes, num_technician_types, max_route_id,timelimit1)
                if obj >1:
                    df_routes = update_routes_df_and_csv(new_routes, df_routes, route_File)
                    max_route_id = updated_max_route_id  # Update max_route_id for the next iteration
                    improvement_found = True  # An improvement has been found
                    improve_count +=1
                if not optimal:
                    optimality = False

        if (not improvement_found and not optimality and improve_count<1):
          improvement_found=  checkwotime(nodes, arcs, dual_prices, n, t, v, df_routes, num_technician_types, max_route_id,timelimit2)


                    

    # Final optimization with the best set of routes
   
   #her må du fiksa slik at du skiller mellom tom for tid og optiml funnet

    print("Final optimization with the best set of routes(LP):")
    model, route_vars = build_master_problem(df_routes, df_parameters)
    lp_solution = optimize_master_problem(model, df_routes, route_vars)

    print("Final optimization with the best set of routes(IP):")
    model_IP, route_vars = build_master_problem_IP(df_routes, df_parameters)
    ip_solution = optimize_master_problem(model_IP, df_routes, route_vars)

    end_time = time.time()  # Capture the end time after function execution
    duration = end_time - start_time

    # Append the solutions to the CSV file
    append_solution_to_csv(df_routes, lp_solution, ip_solution, route_File, duration)












time_limit1 = 10
time_limit2 = 20




Fleets = {1:3,
          2:2,
          3:2,
         } #key = fleet value = nr of vessels


def main():
    for file in data_files:
        for fleets in Fleets:
            global data_file
            data_file = file
    
            global V
            V = Fleets[fleets]

            global Fleet
            Fleet = fleets
            global route_File
            route_File  = f"{data_file}_fleet{fleets}.csv"
            
            global dfs
            dfs= load_data_sub()
            initialize_globals()


            start_time = time.time()  # Capture the start time
            run_column_generation(time_limit1, time_limit2, route_File)
            end_time = time.time()  # Capture the end time after function execution
            duration = end_time - start_time  # Calculate the duration
            print(f"The run_column_generation function took {duration} seconds to complete.")





if __name__ == "__main__":
    main()


    
    