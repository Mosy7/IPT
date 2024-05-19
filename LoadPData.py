import pandas as pd
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader

# Load dataset (replace with your dataset path)
data_path = 'path/to/your/dataset.csv'
data = pd.read_csv(data_path)

# Display the first few rows of the dataset
print(data.head())

# Assume the dataset has columns: user_id, item_id, rating
# Convert to surprise format
reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)

# Split data into training and test sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
