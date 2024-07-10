#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
Data=pd.read_csv('Iris.csv')
print(Data)


# In[32]:


import pandas as pd
Data = pd.read_csv("Iris.csv")

print("Column with null values:")
for i in Data.columns:
    if Data[f"{i}"].isna().sum()>0:
        print("{}".format(i))
mean_spl = Data['SepalLengthCm'].mean()
print(Data['SepalLengthCm'].fillna(mean_spl, inplace=True))

mean_swl = Data['SepalWidthCm'].mean()
print(Data['SepalWidthCm'].fillna(mean_swl, inplace=True))

var_pl = Data['PetalLengthCm'].var()
print(Data['PetalLengthCm'].fillna(var_pl, inplace=True))

mean_pw = Data['PetalWidthCm'].mean()
print(Data['PetalWidthCm'].fillna(mean_pw, inplace=True))


# In[33]:


Data.to_csv("hilda.csv")


# In[34]:


import pandas as pd
data=pd.read_csv("hilda.csv")
data


# In[35]:


import matplotlib.pyplot as plt
plt.boxplot(data["SepalLengthCm"])


# In[36]:


plt.boxplot(data["SepalWidthCm"])


# In[37]:


import numpy as np
z_scores=np.abs((data-data.mean()/data.std()))
threshold=3
outliers=(z_scores>threshold).any(axis=1)
print(data[outliers])


# In[38]:


Data.shape


# In[39]:


Data.shape[0]


# In[40]:



Data.shape[1]


# In[41]:


Data.nunique()


# In[42]:


#prifilinge is for analysis and removing null values


# In[43]:


#import pandas_profiling as pp
pp.ProfileReport(Data)


# In[ ]:


Data.isnull().sum()


# In[ ]:


Data.isna().sum()


# In[ ]:


Data.mode()


# In[ ]:


Data.median()


# In[ ]:


Data.mean()


# In[ ]:


Data.std()


# In[ ]:


Data.var()


# In[ ]:


import pandas as pd
Data = pd.read_csv("Iris.csv")

print("Column with null values:")
for i in Data.columns:
    if Data[f"{i}"].isna().sum()>0:
        print("{}".format(i))
mean_spl = Data['SepalLengthCm'].mean()
print(Data['SepalLengthCm'].fillna(mean_spl, inplace=True))

mean_swl = Data['SepalWidthCm'].mean()
print(Data['SepalWidthCm'].fillna(mean_swl, inplace=True))

var_pl = Data['PetalLengthCm'].var()
print(Data['PetalLengthCm'].fillna(var_pl, inplace=True))

mean_pw = Data['PetalWidthCm'].mean()
print(Data['PetalWidthCm'].fillna(mean_pw, inplace=True))


# In[ ]:


from ydata_profiling import ProfileReport
ProfileReport(Data)


# In[ ]:


for i Data.columns:{
    if Data[f"{i}"].isna().sum()>O
    print


# In[ ]:


import numpy as np

# Define the list of data points
data_points = [10, 386, 479, 627, 20, 523, 482, 483, 542, 699, 535, 617, 577, 987]

# Calculate the mean of the data points
mean = np.mean(data_points)

# Calculate the standard deviation of the data points
std_dev = np.std(data_points)

# Define the threshold for identifying outliers
threshold = 2  # Typically, 2 or 3 standard deviations are used

# Identify and remove outliers
filtered_data_points = [x for x in data_points if (mean - threshold * std_dev) <= x <= (mean + threshold * std_dev)]

# Display the new list without the outliers
print("Original data points:", data_points)
print("Filtered data points:", filtered_data_points)


# # TB MODEL

# In[ ]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the model parameters
beta = 0.3   # Transmission rate
sigma = 0.1  # Rate of progression from exposed to infectious
gamma = 0.05 # Recovery rate
mu = 0.01    # Natural mortality rate
delta = 0.02 # Disease-induced mortality rate

# Define the initial conditions
S0 = 990  # Initial number of susceptible individuals
E0 = 10   # Initial number of exposed individuals
I0 = 5    # Initial number of infectious individuals
R0 = 0    # Initial number of recovered individuals

# Total population, N
N = S0 + E0 + I0 + R0

# A grid of time points (in days)
t = np.linspace(0, 160, 160)

# The SEIR model differential equations
def deriv(y, t, N, beta, sigma, gamma, mu, delta):
    S, E, I, R = y
    dSdt = mu * N - beta * S * I / N - mu * S
    dEdt = beta * S * I / N - sigma * E - mu * E
    dIdt = sigma * E - gamma * I - mu * I - delta * I
    dRdt = gamma * I - mu * R
    return dSdt, dEdt, dIdt, dRdt
print()

# Initial conditions vector
y0 = S0, E0, I0, R0

# Integrate the SEIR equations over the time grid, t
ret = odeint(deriv, y0, t, args=(N, beta, sigma, gamma, mu, delta))
S, E, I, R = ret.T

# Plot the data
fig = plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
plt.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Exposed')
plt.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infectious')
plt.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('TB Model (SEIR)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


for number in range(1, 101):
    square = number ** 2
    print(f"The square of {number} is {square}")


# In[ ]:


# Function to check if a number is prime
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

# Generate and print squares of prime numbers from 1 to 100
for number in range(1, 101):
    if is_prime(number):
        square = number ** 2
        print(f"The square of prime number {number} is {square}")


# In[ ]:


import pandas as pd

# Create first DataFrame using dictionary and Pandas Series
data1 = {
    'ID': pd.Series([1, 2, 3, 4, 5]),
    'Name': pd.Series(['Alice', 'Bob', 'Charlie', 'David', 'Eva'])
}
df1 = pd.DataFrame(data1)

# Create second DataFrame using dictionary and Pandas Series
data2 = {
    'ID': pd.Series([1, 2, 3, 6]),
    'Age': pd.Series([25, 30, 35, 40]),
    'City': pd.Series(['New York', 'Los Angeles', 'Chicago', 'Houston'])
}
df2 = pd.DataFrame(data2)

# Merge the DataFrames on 'ID' column
merged_df = pd.merge(df1, df2, on='ID', how='inner')

# Display the merged DataFrame
print("Merged DataFrame:")
print(merged_df)


# In[ ]:


import pandas as pd

# Create first DataFrame using dictionary and Pandas Series
data1 = {
    'ID': pd.Series([1, 2, 3, 4, 5]),
    'Name': pd.Series(['Alice', 'Bob', 'Charlie', 'David', 'Eva'])
}
df1 = pd.DataFrame(data1)

# Create second DataFrame using dictionary and Pandas Series
data2 = {
    'ID': pd.Series([1, 2, 3, 6]),
    'Age': pd.Series([25, 30, 35, 40]),
    'City': pd.Series(['New York', 'Los Angeles', 'Chicago', 'Houston'])
}
df2 = pd.DataFrame(data2)

# Inner Join
inner_join_df = pd.merge(df1, df2, on='ID', how='inner')
print("Inner Join DataFrame:")
print(inner_join_df)

# Left Join
left_join_df = pd.merge(df1, df2, on='ID', how='left')
print("\nLeft Join DataFrame:")
print(left_join_df)

# Right Join
right_join_df = pd.merge(df1, df2, on='ID', how='right')
print("\nRight Join DataFrame:")
print(right_join_df)

# Full Join (Outer Join)
full_join_df = pd.merge(df1, df2, on='ID', how='outer')
print("\nFull Join (Outer Join) DataFrame:")
print(full_join_df)


# In[44]:


import pandas as pd
import numpy as np

# Sample data
data = {
    'ID': pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    'Value': pd.Series([10, 12, 14, 15, 18, 20, 21, 22, 23, 100])  # The value 100 is an outlier
}
df = pd.DataFrame(data)

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df['Value'].quantile(0.25)
Q3 = df['Value'].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Determine the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[(df['Value'] < lower_bound) | (df['Value'] > upper_bound)]

# Print results
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
print("Outliers:")
print(outliers)


# In[ ]:




