# Project 3: AI Recommendation Logic

A simple movie recommendation system that matches user preferences to items using **cosine similarity**.

## 🎯 Goal
Create a simple recommendation system based on user preferences.

## 📊 Dataset
A small curated set of **15 movies**, each tagged with one or more genres (e.g., Sci-Fi, Action, Comedy, Romance) spanning **12 unique genres**.

## 🧠 Algorithm
**Cosine Similarity** between a multi-hot genre vector for the user's stated preferences and each movie's genre vector — a common pattern-matching technique used in real-world recommender systems.

## 🔑 Key Requirements
1. **Take user input** — the user types their favorite genres into the terminal (comma-separated)
2. **Vectorize preferences and items** — genres are converted into multi-hot vectors
3. **Match preferences using similarity** — cosine similarity scores each movie against the user's profile
4. **Display recommended items** — the top 5 highest-scoring movies are shown, ranked by match score

## 🛠️ Key Skills Demonstrated
- Logic building
- Pattern matching
- Recommendation concepts (vector-space similarity)

## 📁 Files
| File | Description |
|------|-------------|
| `project3_ai_recommendation.py` | Main script (Jupyter-style, interactive terminal input) |

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn
```

### Run the project
```bash
python project3_ai_recommendation.py
```

You'll be prompted to enter your favorite genres — try something like `Sci-Fi, Action` or `Comedy, Animation`.

Available genres: `Action, Adventure, Animation, Comedy, Crime, Drama, Fantasy, Horror, Musical, Romance, Sci-Fi, Thriller`

Or open it in **Jupyter**, **VS Code**, or **JupyterLab** — the `# %%` markers split it into interactive notebook-style cells.

## 📈 Example Run
```
Enter your favorite genres, separated by commas: Sci-Fi, Action

Top 5 recommendations for you:

1. The Matrix           (genres: Sci-Fi, Action | match score: 1.00)
2. Inception             (genres: Sci-Fi, Thriller | match score: 0.50)
3. Interstellar          (genres: Sci-Fi, Drama | match score: 0.50)
4. Mad Max: Fury Road     (genres: Action, Adventure | match score: 0.50)
5. A Quiet Place         (genres: Horror, Sci-Fi | match score: 0.50)
```

## 💡 How It Works
Each movie's genres are converted into a **multi-hot vector** (a 1 for each genre it has, 0 otherwise) across all genres in the dataset. The user's stated preferences are converted into the same kind of vector. **Cosine similarity** then measures the angle between the user's vector and each movie's vector — the closer the vectors point in the same direction, the higher the match score. Movies are ranked by this score, and the top matches are recommended.

## 📄 License
This project is open source and available for educational use.
