#Import files from directory.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from ast import literal_eval
from sklearn.preprocessing import LabelEncoder

#for weights
from sklearn.inspection import permutation_importance
import numpy as np


#for AUC ROC libraries
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# JSON file path
json_file = 'output_sen_vec.json'

df = pd.read_json(json_file)

# df contains JSON data as a DataFrame
print(df.head())  # Print the first few rows of the DataFrame

print(df.columns)

data = df[['used type', 'embedded.posts.comments.text', 'posts.likes_count', 'pot.comment.likes_count', 'sentiment', 'popularity']]

# Drop rows with NaN values if any
data = data.dropna()

# Convert string representation of lists to actual lists of floats
#data['embedded.posts.comments.text'] = data['embedded.posts.comments.text'].apply(literal_eval)

# Convert the 'embedded.posts.comments.text' column into separate columns
max_length = max(len(x) for x in data['embedded.posts.comments.text'])
comment_columns = pd.DataFrame(data['embedded.posts.comments.text'].to_list(),
                               columns=[f'embedded.posts.comments.text_{i}' for i in range(max_length)])

comment_columns.index = data.index
data = pd.concat([data.drop(columns=['embedded.posts.comments.text']), comment_columns], axis=1)

# One-hot encode the 'sentiment' column if it's categorical
encoder = OneHotEncoder(drop='first')
sentiment_encoded = encoder.fit_transform(data[['sentiment']]).toarray()
sentiment_df = pd.DataFrame(sentiment_encoded, columns=encoder.get_feature_names_out(['sentiment']))

# Concatenate the one-hot encoded sentiment column back to x
data = pd.concat([data.drop(columns=['sentiment']), sentiment_df], axis=1)

print(data['used type'].head())

label_encoder = LabelEncoder()
data['used type'] = label_encoder.fit_transform(data['used type'])

print(data['used type'].head())


print(data.isna().sum())
print(f'data_len: {len(data)}')

# Drop rows with NaN values if any
data = data.dropna()
print(f'data_dropNa_len: {len(data)}')

y = data['used type']
x = data.drop(columns=['used type'])

print(f'x: {len(x)}')
print(f'y: {len(y)}')

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f'X_train: {len(X_train)}')
print(f'X_test: {len(X_test)}')
print(f'y_train: {len(y_train)}')
print(f'y_test: {len(y_test)}')

# Normalize the feature columns
print('start StandardScaler')
scaler = StandardScaler()
print('finish StandardScaler')
print('start scaler.fit_transform(X_train)')
X_train = scaler.fit_transform(X_train)
print('finish scaler.fit_transform(X_train)')
print('start scaler.transform(X_test)')
X_test = scaler.transform(X_test)
print('finish scaler.transform(X_test)')

# Create a neural network model
model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=500, random_state=42)

print('start model.fit')
# Fit the model on the training data
model.fit(X_train, y_train)
print('finish model.fit')
print('start model.predict')
# Predict on test data
y_pred = model.predict(X_test)
print('finish model.predict')
# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print classification report
print(classification_report(y_test, y_pred))

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)




# Calculate and print AUC
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f'AUC: {auc:.2f}')

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Extract and print model weights
print("\nModel Weights:")
#Import files from C:\seminar



#Retrieve weights-NO NEED
# weights = model.coefs_
# for i, weight_matrix in enumerate(weights):
#     print(f'Layer {i} weights:')
#     print(weight_matrix)
# # Compute permutation feature importance
# perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
#
# # Display feature importances
# for i in perm_importance.importances_mean.argsort()[::-1]:
#     if perm_importance.importances_mean[i] - 2 * perm_importance.importances_std[i] > 0:
#         print(f"Feature: {x.columns[i]}, Importance: {perm_importance.importances_mean[i]:.4f} +/- {perm_importance.importances_std[i]:.4f}")
