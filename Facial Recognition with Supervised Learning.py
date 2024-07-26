# Import required libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file 
df = pd.read_csv("D:\Machine Learning\Facial Recognition with Supervised Learning\dataset\lfw_arnie_nonarnie.csv")

# Seperate the predictor and class label
X = df.drop('Label', axis=1)
y = df['Label'] 

# Split the data into training and testing sets using stratify to balance the class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)
# Start coding here
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

models = {
    'LogisticRegression': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ]),
    'SVC': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(probability=True, random_state=42))
    ]),
    'RandomForest': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
}
param_grid = {
    'LogisticRegression': {'classifier__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]},  # Expanded range
    'SVC': {'classifier__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 'classifier__gamma': ['scale', 'auto', 0.1, 1, 10]},  # Expanded range and added gamma options
    'RandomForest': {'classifier__n_estimators': [50, 100, 200, 500]}  # Increased n_estimators
}


best_model_name = None
best_model_info = {}
best_model_cv_score = 0

for name, model in models.items():
    grid_search = GridSearchCV(model, param_grid[name], cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    if grid_search.best_score_ > best_model_cv_score:
        best_model_name = name
        best_model_info = grid_search.best_params_
        best_model_cv_score = grid_search.best_score_

print(f"Best Model: {best_model_name}")
print(f"Best Model Info: {best_model_info}")
print(f"Best Model CV Score: {best_model_cv_score}")

best_model = models[best_model_name]
best_model.set_params(**best_model_info)
best_model.fit(X_train, y_train)

predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()