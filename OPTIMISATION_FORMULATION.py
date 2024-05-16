# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import required libraries
import pandas as pd
import numpy as np
import sklearn.neighbors
from pulp import *

#read datasets
customer_location = pd.read_excel('DatasetFinal.xlsx', sheet_name=16)

facility_location = pd.read_excel('DatasetFinal.xlsx', sheet_name=17)

Customer_demand = pd.read_excel('DatasetFinal.xlsx', sheet_name=18)

facility_capacities = pd.read_excel('DatasetFinal.xlsx', sheet_name=19)

cost = pd.read_excel('DatasetFinal.xlsx', sheet_name=20)

#old_allocation = solution from Q3 constructive heuristic algorithm
old_allocations = pd.read_excel('DatasetFinal.xlsx', sheet_name=2)


#convert old_allocations to arrays and save in variable A
old_allocations = old_allocations.to_numpy()
old_allocations = old_allocations[:,1:]
A = old_allocations

#convert cost dataframe to array
cost = cost.to_numpy()
cost = cost[:,1:]



# Define the problem
problem = LpProblem("Assignment Problem", LpMinimize)

# Define the decision variables
num_customers = 363
num_facilities = 13
x = LpVariable.dicts("x", [(i,j) for i in range(num_customers) for j in range(num_facilities)], cat=LpBinary)
r = LpVariable.dicts("r", [i for i in range(num_customers)], cat=LpBinary)

# Define the cost matrix
cost_matrix = cost * 2 #round trip

# Define the customer demand and facility capacity arrays
h=Customer_demand['CUSTOMER DEMAND'].tolist()
H=np.array(h)


c=[4000,3000,13000,10000,13000,16000,2000,2000,3000,3000,4000,3000,13000]
C=np.array(c)


# Define the objective function
problem += lpSum([H2[i] * cost_matrix[i][j] * x[(i,j)] for i in range(num_customers) for j in range(num_facilities)])

# Define the constraints
for i in range(num_customers):
    problem += lpSum([x[(i,j)] for j in range(num_facilities)]) == 1 #ensures all demands are served and one customer is allocated to one facility only

for j in range(num_facilities):
    problem += lpSum([H2[i] * x[(i,j)] for i in range(num_customers)]) <= C[j] #ensures demands in each new allocation does not exceed the corresponding facility capacity

for i in range(num_customers):
    problem += r[i] == 1 - lpSum([x[(i,j)] * A[i][j] for j in range(num_facilities)]) #indicates whether customer i is reallocated

#ensures a customer is reallocated only when the reallocation savings is greater than the reallocation cost
#reallocation cost = 6 hours
problem += lpSum([H2[i] * cost_matrix[i][j] * (A[i][j] - x[(i,j)]) for i in range(num_customers) for j in range(num_facilities)]) >= lpSum([r[i] * 6 for i in range(num_customers)])

# Solve the problem
problem.solve()

# Print the solution
for i in range(num_customers):
    for j in range(num_facilities):
        if x[(i,j)].varValue == 1:
            print("Customer", i, "is allocated to Facility", j)

print("Total Cost:", value(problem.objective))


# Save the solution as an array
solution_array = [[x[(i,j)].varValue for j in range(num_facilities)] for i in range(num_customers)]

new_allocations=np.array(solution_array)

new_allocations_df = (pd.DataFrame(new_allocations, index=customer_location['Name'], columns=facility_location['Name']))







