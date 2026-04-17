import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

import pickle

# Load data

data = pd.read_csv('data/student_data.csv')

# Features & target

X = data[['score', 'accuracy', 'time_taken', 'attempts']]

y = data['rank']

# Split data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model

model = RandomForestRegressor()

model.fit(X_train, y_train)

# Save model

with open('model/model.pkl', 'wb') as f:

    pickle.dump(model, f)

print("Model trained and saved!")