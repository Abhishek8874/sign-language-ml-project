import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# This script trains a machine learning model to recognize hand signs.

# 1. LOAD DATA
# Load the data processed by create_dataset_mediapipe.py
# The data contains lists of 84 numbers (hand coordinates) and their labels (A, B, C...)
try:
    data_dict = pickle.load(open('./data.pickle', 'rb'))
except FileNotFoundError:
    print("Error: 'data.pickle' not found. Run create_dataset_mediapipe.py first.")
    exit()

# Convert list of lists to a numpy array for efficient processing
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# 2. SPLIT DATA
# Separation: 80% for training (learning), 20% for testing (checking accuracy)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# 3. INITIALIZE MODEL
# We use a Random Forest Classifier. It's like a committee of decision trees.
# It's robust and works well with this kind of coordinate data.
model = RandomForestClassifier()

print("Training model...")

# 4. TRAIN MODEL
# Feed the training data (x_train) and answers (y_train) to the model.
model.fit(x_train, y_train)

# 5. TEST ACCURACY
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

# 6. SAVE MODEL
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

print("Model saved to 'model.p'")
