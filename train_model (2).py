from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import joblib

# Generate sample data
X, y = make_classification(
    n_samples=500,
    n_features=4,
    n_informative=3,
    n_redundant=1,
    random_state=42
)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, "churn_model.pkl")

print("Model trained successfully!")
print("churn_model.pkl saved successfully!")
