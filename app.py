from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

app = FastAPI()

# Load dataset
iris = load_iris()

# Train model (KNN)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(iris.data, iris.target)

# Class labels
class_names = ["setosa", "versicolor", "virginica"]

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}

# Prediction endpoint
@app.get("/predict")
async def predict(sl: float, sw: float, pl: float, pw: float):
    features = np.array([[sl, sw, pl, pw]])
    pred = int(model.predict(features)[0])
    
    return {
        "prediction": pred,
        "class_name": class_names[pred]
    }