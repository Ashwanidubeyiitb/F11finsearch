import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def index1(age):
    if age < 20:
        score1 = 0
    elif age >= 20 and age < 30:
        score1 = 0.8
    elif age >= 30 and age < 40:
        score1 = 1
    elif age >= 40 and age < 50:
        score1 = 0.6
    elif age >= 60:
        score1 = 0.4
    else:
        score1 = None  # Handle invalid age
    
    return score1

def score2(education):
    if education == 1:
        score2 = 0.3
    elif education == 2:
        score2 = 0.6
    elif education == 3:
        score2 = 0.9
    else:
        score2 = None  # Handle invalid education level
    
    return score2

# Load data from CSV file
csv_path = "C://Users//Ashwani//Downloads//mm225tut//tut2//tut3//assignment//assignment1//UCI_Credit_Card.csv"  # Replace with your CSV file path
data = pd.read_csv(csv_path)

# Calculate scores for each age
scores1 = []
for age in data["AGE"]:
    score = index1(age)
    scores1.append(score)
data["score1"] = scores1

# Calculate scores for each education level
scores2 = []
for education in data["EDUCATION"]:
    score = score2(education)
    scores2.append(score)
data["score2"] = scores2

# Calculate score3 (frequency probability of zero payments)
payment_columns = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
data['score3'] = (data[payment_columns] == 0).sum(axis=1) / len(payment_columns)

data['credit_score'] = data['score1'] + data['score2'] + data['score3']

for index, row in data.iterrows():
    print(f"Entry {index}: Age: {row['AGE']} | Education: {row['EDUCATION']} | Credit Score: {row['credit_score']:.2f}")

# Split the data into features (X) and target (y)
data.dropna(inplace=True)
X = data.drop(['default.payment.next.month'], axis=1)

y = data['default.payment.next.month']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
pd
# Build and train the random forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on the testing data
y_pred = model.predict(X_test_scaled)

for index, prediction in enumerate(y_pred):
    print(f"Index: {index} | Predicted Value: {prediction}")

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print model performance metrics
print("\nRandom Forest Model Performance:")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

# Plot credit score vs default.payment.next.month
# ...

# ...

# Plot credit score vs default.payment.next.month with color-coded predictions
sample_size = len(y_pred)  # Choose the number of data points to plot
plt.figure(figsize=(10, 6))
plt.scatter(data['credit_score'][:sample_size], data['default.payment.next.month'][:sample_size], c=y_pred, cmap='coolwarm', alpha=0.6)
plt.title("Credit Score vs Default Payment")
plt.xlabel("Credit Score")
plt.ylabel("Default Payment")
plt.grid(True)
plt.show()
