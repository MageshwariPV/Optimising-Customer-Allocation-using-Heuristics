#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 19:24:39 2023

@author: anitachinye
"""

#import required libraries
import pandas as pd
import numpy as np
import sklearn.neighbors
import random

#read datasets
customer_location = pd.read_excel('DatasetFinal4.xlsx', sheet_name=0)
facility_location = pd.read_excel('DatasetFinal4.xlsx', sheet_name=1, index_col=0)
Customer_demand = pd.read_excel('DatasetFinal4.xlsx', sheet_name=2)

h=Customer_demand['CUSTOMER DEMAND'].tolist()#convert customer demand to list
H=np.array(h)#convert list to array and save as new object

#add new columns with longitude and latitude values converted to radians
customer_location[['lat_radians','long_radians']] = (np.radians(customer_location.loc[:,['LAT','LONG']]))
facility_location[['lat_radians','long_radians']] = (np.radians(facility_location.loc[:,['LAT','LONG']]))


#create a matrix that contains the distance between customer locations and facility locations in Kilometres.
dist = sklearn.neighbors.DistanceMetric.get_metric('haversine')
distance_matrix = (dist.pairwise(customer_location[['lat_radians','long_radians']], facility_location[['lat_radians','long_radians']])*6371)


#convert matrix to pandas dataframe
df_distance_matrix = pd.DataFrame(distance_matrix, index=customer_location['Name'], columns=facility_location.index)



#CONSTRUCTIVE HEURISTICS (GREEDY RANDOM ALGORITHM)
def randomised_greedy(customer_location, facility_location, k, seed=10):
    # Initialize allocation matrix
    allocation_matrix = np.zeros((len(customer_location), len(facility_location)), dtype=int)
    allocation_matrix = pd.DataFrame(allocation_matrix, index=customer_location['Name'], columns=facility_location.index)

    # Initialize random number generator
    if seed is not None:
        random.seed(seed)

    # Allocate customers to facilities
    shuffled_indices = list(customer_location.index)
    random.shuffle(shuffled_indices)
    for i in shuffled_indices:
        # Compute distances to facilities
        distances = df_distance_matrix.iloc[shuffled_indices[i]]
        # Sort distances from nearest to farthest
        distances = distances.sort_values()
        # Randomly select between the nearest facility and the kth-nearest facility
        j_range = range(min(len(facility_location), k))
        j = min(j_range, key=lambda x: distances.iloc[x] + random.random())
        # Find first facility with enough capacity
        while facility_location.loc[distances.index[j],'Capacity'] < customer_location.loc[shuffled_indices[i], 'CUSTOMER DEMAND']:
            j += 1
            if j not in j_range:
                break
        # Allocate customer to facility
        if j in j_range and facility_location.loc[distances.index[j],'Capacity'] >= customer_location.loc[shuffled_indices[i], 'CUSTOMER DEMAND']:
            allocation_matrix.loc[customer_location.loc[shuffled_indices[i], 'Name'], distances.index[j]] = 1
            facility_location.loc[distances.index[j],'Capacity'] -= customer_location.loc[shuffled_indices[i], 'CUSTOMER DEMAND']

    return allocation_matrix


# Call the randomised_greedy function
allocation_matrix = randomised_greedy(customer_location, facility_location, k=13, seed=10)

# Print the allocation matrix
print(allocation_matrix)

costs = distance_matrix/60 #convert distance matrix to time matrix(time = distance/speed)
costs = costs*2#account for round trip

allocation_matrix_array = allocation_matrix.to_numpy() #convert solution from constructive heuristics to an array from a dataframe


#calculate total time taken
minimum_time = 0

for (l, m), cost_value in np.ndenumerate(costs):
    allocation_value = allocation_matrix_array[l, m]
    minimum_time += cost_value * allocation_value * H[l]
    
print(minimum_time)





#FIRST IMPROVEMENT LOCAL SEARCH ALGORITHM

#define compute_obj_value function
def compute_obj_value(allocation_array, customer_location, facility_location):
    # Compute the total distance travelled
    distance_travelled = 0
    for i in range(allocation_array.shape[0]):
        for j in range(allocation_array.shape[1]):
            if allocation_array[i,j] == 1:
                distance_travelled += df_distance_matrix.iloc[i,j]
    # Compute the total cost of unfulfilled demand
    unfulfilled_demand_cost = 0
    for i in range(allocation_array.shape[0]):
        if allocation_array[i,:].sum() == 0:
            unfulfilled_demand_cost += customer_location.loc[customer_location.index[i],'CUSTOMER DEMAND']
    return distance_travelled + unfulfilled_demand_cost



     
def first_improvement_local_search(customer_location, facility_location, allocation_matrix, reallocation_cost=6):
    # Convert the allocation matrix to a numpy array
    allocation_array = allocation_matrix.to_numpy()

    # Compute the initial objective value
    obj_value = compute_obj_value(allocation_array, customer_location, facility_location)
    print('Initial objective value:', obj_value)

    # Initialize the best solution
    best_allocation = allocation_array.copy()
    best_obj_value = obj_value

    # Loop over all pairs of customer-facility assignments
    for i in range(allocation_array.shape[0]):
        for j in range(allocation_array.shape[1]):
            # Skip if the customer is already assigned to the facility
            if allocation_array[i][j] == 1:
                continue

            # Compute the cost of reassigning the customer to the facility
            old_facility_index = np.argmax(allocation_array[i])
            old_distance = df_distance_matrix.iloc[i][facility_location.index[old_facility_index]]
            new_distance = df_distance_matrix.iloc[i][facility_location.index[j]]
            cost = new_distance - old_distance

            # Check if the new allocation would result in a better objective value
            if cost + reallocation_cost < 0:
                # Update the allocation matrix and the objective value
                new_allocation = allocation_array.copy()
                new_allocation[i][old_facility_index] = 0
                new_allocation[i][j] = 1

                # Compute the new objective value
                new_obj_value = compute_obj_value(new_allocation, customer_location, facility_location)

                # Update the best solution if necessary
                if new_obj_value < best_obj_value:
                    best_allocation = new_allocation.copy()
                    best_obj_value = new_obj_value

    # Convert the best allocation to a numpy array
    best_allocation_array = np.array(best_allocation)

    # Print the best allocation array
    print('Best allocation array:', best_allocation_array)

    # Return the best allocation matrix as a pandas DataFrame
    return pd.DataFrame(best_allocation, index=customer_location.index, columns=facility_location.index)
      

# Call the function and assign the result to a variable
best_allocation_array = first_improvement_local_search(customer_location, facility_location, allocation_matrix)

# Print the variable
print(best_allocation_array)



array_best_allocation = best_allocation_array.to_numpy()


#calculate total travel time 
minimum_time = 0

for (l, m), cost_value in np.ndenumerate(costs):
    allocation_value = array_best_allocation[l, m]
    minimum_time += cost_value * allocation_value * H[l]
    
print(minimum_time)











