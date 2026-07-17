# AI Projects Portfolio

A collection of beginner-friendly AI/ML projects covering classification and recommendation systems.

---

## Project 2: Data Classification Using AI

A machine learning project that builds a classification model on a small dataset using **K-Nearest Neighbors (KNN)**.

### 🎯 Goal
Build a basic classification model using a small dataset, covering the full workflow from data loading to model evaluation.

### 📊 Dataset
**Iris Flower Dataset** — a classic, built-in dataset (via `scikit-learn`) containing 150 samples of iris flowers across 3 species (*setosa*, *versicolor*, *virginica*), described by 4 features:
- Sepal length
- Sepal width
- Petal length
- Petal width

### 🧠 Algorithm
**K-Nearest Neighbors (KNN)** with `k=5`, using standardized features for distance-based classification.

### 🔑 Key Steps
1. **Load and understand the dataset** — inspect shape, class distribution, and summary statistics
2. **Split data into training and testing sets** — 80% train / 20% test, stratified by class
3. **Apply a classification algorithm** — scale features and train a KNN model
4. **Evaluate the model** — accuracy score, classification report, and confusion matrix
5. **Make predictions** — classify a new, unseen sample

### 🛠️ Key Skills Demonstrated
- Data handling with `pandas`
- Supervised learning basics
- Model training and evaluation with `scikit-learn`

### 📈 Sample Results
On a typical run, the model achieves **~93% accuracy** on the test set, with strong precision/recall across all three species.

### ▶️ Run it
```bash
python project2_data_classification.py
```

---

## Project 3: AI Recommendation Logic

A simple movie recommendation system that matches user preferences to items using **cosine similarity**.

### 🎯 Goal
Create a simple recommendation system based on user preferences.

### 📊 Dataset
A small curated set of **15 movies**, each tagged with one or more genres (e.g., Sci-Fi, Action, Comedy, Romance) spanning **12 unique genres**.

### 🧠 Algorithm
**Cosine Similarity** between a multi-hot genre vector for the user's stated preferences and each movie's genre vector — a common pattern-matching technique used in real-world recommender systems.

### 🔑 Key Steps
1. **Take user input** — the user types their favorite genres into the terminal (comma-separated)
2. **Vectorize preferences and items** — genres are converted into multi-hot vectors
3. **Match preferences using similarity** — cosine similarity scores each movie against the user's profile
4. **Display recommended items** — the top 5 highest-scoring movies are shown, ranked by match score

### 🛠️ Key Skills Demonstrated
- Logic building
- Pattern matching
- Recommendation concepts (vector-space similarity)

### ▶️ Run it
```bash
python project3_ai_recommendation.py
```
You'll be prompted to enter your favorite genres — try something like `Sci-Fi, Action` or `Comedy, Animation`.

---

## 📁 Files
| File | Description |
|------|-------------|
| `project2_data_classification.py` | Classification project (Jupyter-style, runs as `.py` or notebook) |
| `project3_ai_recommendation.py` | Recommendation project (Jupyter-style, interactive terminal input) |

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas scikit-learn numpy
```

Both scripts use `# %%` cell markers, so they can be opened as interactive notebooks in **Jupyter**, **VS Code**, or **JupyterLab**, or run directly from the command line.

## 📄 License
This project is open source and available for educational use.
