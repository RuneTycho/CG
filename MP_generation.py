import pandas as pd

# Define the dummy data
data = {
    "Route ID": [1, 2, 3, 4, 5, 6, 7, 8],
    "Vessel": ["Vessel_1", "Vessel_1", "Vessel_2", "Vessel_2", "Vessel_3", "Vessel_3", "Vessel_4", "Vessel_4"],
    "Time Window": [1, 2, 1, 3, 2, 3, 1, 2],
    "Profit": [20, 2, -0, 40, 15, 5, 25, 10],
    "Technician Type 1 Demand": [2, 1, 1, 2, 1, 2, 2, 1],
    "Technician Type 2 Demand": [1, 2, 1, 2, 2, 1, 1, 2],
    "Jobs Included": ["Job_1 Job_2", "Job_3", "Job_4 Job_5", "Job_6", "Job_7", "Job_8 Job_9", "Job_10", "Job_11 Job_12"]
}


# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("dummy_routes.csv", index=False)
