# ML-ASSIGNMENT-DEVANSH
# Data Analysis and Modeling Examples

This repository contains examples of data analysis and modeling using Python. It includes code snippets for working with datasets, splitting data, training models, and evaluating performance. The examples use the Iris dataset and a sample dataset for linear regression.

## 1. Iris Dataset Analysis

This script demonstrates how to load and explore the Iris dataset using `scikit-learn` and `pandas`. The dataset is converted to a DataFrame, and basic statistics and a preview of the dataset are displayed.

### Code

```python
from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()

# Convert to a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Display the first five rows
first_five_rows = df.head()

# Display the dataset’s shape
dataset_shape = df.shape

# Display summary statistics
summary_statistics = df.describe()

# Display the results
print("First five rows of the dataset:\n", first_five_rows)
print("\nShape of the dataset:", dataset_shape)
print("\nSummary statistics:\n", summary_statistics)
pip install pandas scikit-learn

Feel free to adjust the details according to your specific needs or preferences.
