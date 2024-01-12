import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset from a local path
file_path = "/workspaces/Credit-card-Fraud-Detection/fraudTest.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())

# Check if 'Amount' column exists in the dataset
if 'Amount' in df.columns:
    # Remove 'Amount' column
    df = df.drop(['Amount'], axis=1)

    # Split the data into features and target
    X = df.drop(['Class'], axis=1)
    y = df['Class']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
else:
    print("'Amount' column not found in the dataset.")
