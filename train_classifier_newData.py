import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load the data and labels
X = np.load('./data/signs_dataset/X.npy')  # Shape: (2062, 64, 64)
y = np.load('./data/signs_dataset/Y.npy')  

# Flatten the images from 64x64 to 1D (4096 features)
X_flattened = X.reshape(X.shape[0], -1)  # Shape: (2062, 4096)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.2, shuffle=True, stratify=y)

# Initialize the Random Forest model
model = RandomForestClassifier()

# Train the model
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)

# Print the result
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model
with open('./data/sign_model.p', 'wb') as f:
    pickle.dump({'sign_model': model}, f)
