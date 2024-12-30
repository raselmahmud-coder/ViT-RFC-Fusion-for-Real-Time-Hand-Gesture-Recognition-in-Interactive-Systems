import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
data_dict = pickle.load(open('./Datasets/Models/data_sign.pickle', 'rb'))

# Convert file names in labels to numeric class labels
numeric_labels = [int(label.split()[1]) for label in data_dict['labels']]
print("Converted numeric labels:", numeric_labels[:10])  # Check first 10 labels

# Ensure all feature vectors have consistent shape (truncate if necessary)
data = [item[:42] if np.shape(item) == (84,) else item for item in data_dict['data']]
data = np.array(data)
labels = np.array(numeric_labels)

# Validate consistency
assert len(data) == len(labels), "Mismatch between data and labels!"
print("Data and labels are consistent.")

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=10, stratify=labels
)

# Train Random Forest model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate the model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model
with open('./Datasets/Models/model_sign.p', 'wb') as f:
    pickle.dump({'model': model}, f)
