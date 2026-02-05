# %% [markdown]
# # Machine Learning Bootcamp Lab

# %%
# Imports - Libraries needed for data manipulation and ML preprocessing
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
# Make sure to install sklearn in your terminal first!
# Use: pip install scikit-learn
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
from io import StringIO  # For reading string data as file
import requests  # For HTTP requests to download data

# %% [markdown]
# # Step 1
# Read in the datasets: raw data access, copying URL

# %%
# College Completion Dataset
college_url = ("https://raw.githubusercontent.com/UVADS/DS-3021/refs/heads/main/data/cc_institution_details.csv")
college = pd.read_csv(college_url)

# Check the structure of the dataset and see if we have any issues with variable classes (data types)
college.info()

# %%
# Job Placement Dataset
job_url = ("https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv")
job = pd.read_csv(job_url)

job.info()

# %% [markdown]
# # Step 2
# ### College Completion Dataset
# **Question:** Can we predict student success based on the college they attended and its characteristics?
# 
# **Independent Business Metric:** Graduation Rates

# %% [markdown]
# **Data Preparation**

# %%
# Drop unneeded variables / only keep neccesary variables
college = college[['index', 'unitid', 'level', 'control', 'cohort_size', 'state', 'ft_pct']]

# %%
# Convert categorical columns to the 'category' data type
cols = ['state', 'level', 'control']
college[cols] = college[cols].astype('category')
college.dtypes

# %%
# Looking at variables of interest

# level variable (Type of Institution)
print(college.level.value_counts()) # Looks good
print() # for space purposes

# control variable (Type of College)
print(college.control.value_counts()) # Looks good
print()

# %%
# cohort_size variable (Size of Cohort)
print(college.cohort_size.value_counts()) # Needs to be standardized
print()

# %%
# Normalizing cohort_size using Min-Max scaling
cohort_normalized = MinMaxScaler().fit_transform(college[['cohort_size']])
print(cohort_normalized[:10])

# %%
# Plot original distribution
college.cohort_size.plot.density()

# %%
# Plot normalized distribution
pd.DataFrame(cohort_normalized).plot.density()

# %%
# state variable (where the college is located)
print(college.state.value_counts())
# There are too many groups for this categorical variable 

# %%
# Perform grouping on state based on region

# Assigning states to regions
northeast = [
    'Maine', 'New Hampshire', 'Vermont', 'Massachusetts',
    'Rhode Island', 'Connecticut', 'New York',
    'New Jersey', 'Pennsylvania'
]

midwest = [
    'Ohio', 'Michigan', 'Indiana', 'Illinois', 'Wisconsin',
    'Minnesota', 'Iowa', 'Missouri', 'North Dakota',
    'South Dakota', 'Nebraska', 'Kansas'
]

south = [
    'Delaware', 'Maryland', 'Virginia', 'West Virginia',
    'Kentucky', 'North Carolina', 'South Carolina',
    'Tennessee', 'Georgia', 'Florida', 'Alabama',
    'Mississippi', 'Arkansas', 'Louisiana',
    'Texas', 'Oklahoma'
]

west = [
    'Montana', 'Idaho', 'Wyoming', 'Colorado', 'New Mexico',
    'Arizona', 'Utah', 'Nevada', 'California', 'Oregon',
    'Washington', 'Alaska', 'Hawaii'
]

# Complete the grouping by adding it to a new 'region' column in college
college['region'] = (
    college.state
    .apply(lambda x: 'Northeast' if x in northeast
                      else 'Midwest' if x in midwest
                      else 'South' if x in south
                      else 'West')
    .astype('category')
)

# Verify that the grouping worked
print(college.region.value_counts())

# Drop the state column
college = college.drop('state', axis=1)

# %%
# One-hot encoding the categorical columns of interest
college_encoded = pd.get_dummies(college, columns = ['level', 'control'])

# Check the results
college_encoded.head()

# %%
# Create a binary target variable
# Student sucess is operationalized using the percentage of full time students

# Find the median to differentiate between high success and low success
cutoff = college.ft_pct.median()

# 1 = > cutoff (high success), 0 = ≤ cutoff (low success)
college_encoded['s_success'] = pd.cut(college_encoded.ft_pct,
                              bins=[-1, cutoff, 100],
                              labels=[0, 1])

# Verify the new column
college_encoded.head()

# %%
# Calculate the prevalence (percentage of high student success)
prevalence = college_encoded.s_success.value_counts()[1] / len(college_encoded.s_success)
print(f"Baseline/Prevalence: {prevalence:.2f}")

# %%
# Dropping any empty rows in college_encoded
college_clean = college_encoded.dropna()

# Verification
college_clean.isna().sum()
college_clean.head()

# %%
# Data Partitioning

# Split the data into three sets:
len(college_clean) # Number of rows in dataset

# Training (2533 samples): Used to train the model
# Tuning (467 samples): Used to turn hyperparameters
# Test (467 samples): Used for final evaluation only

# %%
# First split: separating out training data 
train, test = train_test_split(
    college_clean,
    train_size = 2533,
    stratify = college_clean.s_success
)

# Verify the split sizes
print(f"Training set shape: {train.shape}")
print(f"Test set shape: {test.shape}")

# %%
# Second split: split remaining data (934 samples) into tuning and testing sets
tune, test = train_test_split(
    test, 
    train_size = 0.5,
    stratify = test.s_success
)

# Training set
print("Training set class distribution:")
print(train.s_success.value_counts())

# Tuning set
print("\nTuning set class distribution:")
print(tune.s_success.value_counts())

# Testing set
print("\nTest set class distribution:")
print(test.s_success.value_counts())

# %% [markdown]
# ### Job Placement Dataset
# **Question:** Can we predict a student’s career success based on their academic performance? 
# 
# **Independent Business Metric:** Salary

# %% [markdown]
# **Data Preparation**

# %%
# Convert categorical columns to the 'category' data type
job_cols = ['degree_t', 'workex', 'status']
job[job_cols] = job[job_cols].astype('category')

# Verify data types
job.dtypes

# %%
# Looking at variables of interest

# ssc_p variable (Secondary Education percentage - 10th Grade)
# Essentially: 10th grade final scores
print(job.ssc_p.value_counts()) 

# let's standardize it 
ssc_standardized = StandardScaler().fit_transform(job[['ssc_p']])
print(ssc_standardized[:10])

# %%
# hsc_p variable (Higher Secondary Education percentage - 12th Grade)
# Essentially: 12th grade final scores
print(job.hsc_p.value_counts()) 

# let's standardize it 
hsc_standardized = StandardScaler().fit_transform(job[['hsc_p']])
print(hsc_standardized[:10])

# %% 
# degree_t variable (Degree)
print(job.degree_t.value_counts()) # Looks good

# %%
# workex variable (Work Experience - yes/no)
print(job.workex.value_counts()) # Looks good

# %% 
# status variable (Job Placement - placed/not placed)
print(job.status.value_counts()) # Looks good

# %%
# One-hot encoding the categorical variables of interest
job_encoded = pd.get_dummies(job, columns = ['degree_t', 'workex', 'status'])

# Check the results
job_encoded.head()

# %% 
# Create a binary target variable
# Student career success is operationalized using their 12th grade exam scores

# 1 = > 65% (high career success), 0 = ≤ 65% (low career success)
job_encoded['success'] = pd.cut(job_encoded.hsc_p, 
                                bins = [-1, 65, 100], 
                                labels = [0, 1])

# Verify the new column
job_encoded.head()

# %%
# Calculate the prevalence (percentage of high student career success)
job_prevalence = job_encoded.success.value_counts()[1] / len(job_encoded.success)
print(f"Baseline/Prevalence: {job_prevalence:.2f}")

# %%
# Remove unneeded columns
job_clean = job_encoded.drop(['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'etest_p', 'specialisation', 'mba_p', 'salary'], axis = 1)
job_clean.head()

# Drop any empty rows in job_clean
job_clean.dropna()

# Verification
job_clean.isna().sum()
job_clean.head()

# %% 
# Data Partitioning

# Split the data into three sets:
len(job_clean) # Number of rows in dataset

# Training (151 samples): Used to train the model
# Tuning (32 samples): Used to turn hyperparameters
# Test (32 samples): Used for final evaluation only

# %%
# First split: separating out training data 
train, test = train_test_split(
    job_clean,
    train_size = 151,
    stratify = job_clean.success
)

# Verify the split sizes
print(f"Training set shape: {train.shape}")
print(f"Test set shape: {test.shape}")

# %%
# Second split: split remaining data (934 samples) into tuning and testing sets
tune, test = train_test_split(
    test, 
    train_size = 0.5,
    stratify = test.success
)

# Training set
print("Training set class distribution:")
print(train.success.value_counts())

# Tuning set
print("\nTuning set class distribution:")
print(tune.success.value_counts())

# Testing set
print("\nTest set class distribution:")
print(test.success.value_counts())

# %% [markdown]
# # Step 3
# ### College Completion Dataset
# This dataset is very comprehensive, with lots of categorical variables that can be used. 
# Instictively, I think the 2 year college programs will have higher graduation rates, 
# just because it is a shorter commitment time. Additionally, regions like the Northeast 
# will likely have a higher graduation rate, and therefore student success, due to the 
# number of prestigous schools located in that area.
#
# We can use the data to address our problem. However, I am concerned about how to calculate 
# the said graduation rates using the existing graduation metrics in the dataset (contains lots of 
# separation between year and transfer).

# %% [markdown]
# ### Job Placement Dataset
# This dataset is much harder to understand, and is a smaller dataset than the College 
# Completion Dataset. Instinctively, I think students with higher exam scores (hsc_p) at 
# the end of their secondary education and have work experience will have higher salaries. 
# This is because scores and work experience tend to reflect their proactiveness and commitment 
# to advancing their career. 
# 
# We can use the data to address our problem. However, I am concerned about the size of the dataset, 
# and whether or not the limited data will allow for a good prediction model.


# %%
