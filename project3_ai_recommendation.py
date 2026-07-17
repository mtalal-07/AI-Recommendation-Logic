# %% [markdown]
# # Project 3: AI Recommendation Logic
# **Goal:** Create a simple recommendation system based on user preferences.
#
# **Domain:** Movies
# **Matching Method:** Cosine Similarity (genre-based)

# %%
# --- Imports ---
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# %% [markdown]
# ## Step 1: Build a Small Movie Dataset
# Each movie is tagged with one or more genres.

# %%
movies = [
    {"title": "The Matrix",            "genres": ["Sci-Fi", "Action"]},
    {"title": "Inception",             "genres": ["Sci-Fi", "Thriller"]},
    {"title": "Interstellar",          "genres": ["Sci-Fi", "Drama"]},
    {"title": "The Dark Knight",       "genres": ["Action", "Thriller", "Crime"]},
    {"title": "Mad Max: Fury Road",    "genres": ["Action", "Adventure"]},
    {"title": "The Notebook",          "genres": ["Romance", "Drama"]},
    {"title": "La La Land",            "genres": ["Romance", "Musical", "Drama"]},
    {"title": "Superbad",              "genres": ["Comedy"]},
    {"title": "The Hangover",          "genres": ["Comedy"]},
    {"title": "Get Out",               "genres": ["Horror", "Thriller"]},
    {"title": "A Quiet Place",         "genres": ["Horror", "Sci-Fi"]},
    {"title": "Toy Story",             "genres": ["Animation", "Comedy", "Adventure"]},
    {"title": "Spirited Away",         "genres": ["Animation", "Fantasy", "Adventure"]},
    {"title": "Gladiator",             "genres": ["Action", "Drama", "Adventure"]},
    {"title": "Pride & Prejudice",     "genres": ["Romance", "Drama"]},
]

df = pd.DataFrame(movies)
print(f"Loaded {len(df)} movies.\n")
print(df)

# %% [markdown]
# ## Step 2: Build a Genre Vector Space
# Convert each movie's genre list into a multi-hot vector so we can compare
# it mathematically to a user's preference vector.

# %%
all_genres = sorted({g for m in movies for g in m["genres"]})
print("All available genres:", all_genres)

def vectorize(genre_list, all_genres):
    """Turn a list of genres into a 0/1 vector over all_genres."""
    return np.array([1 if g in genre_list else 0 for g in all_genres])

movie_vectors = np.array([vectorize(m["genres"], all_genres) for m in movies])

# %% [markdown]
# ## Step 3: Take User Input (Preferences)

# %%
def get_user_preferences(all_genres):
    print("\nAvailable genres:", ", ".join(all_genres))
    raw = input("Enter your favorite genres, separated by commas: ")
    chosen = [g.strip().title() for g in raw.split(",") if g.strip()]
    valid = [g for g in chosen if g in all_genres]
    invalid = [g for g in chosen if g not in all_genres]
    if invalid:
        print(f"(Ignoring unrecognized genres: {', '.join(invalid)})")
    return valid

# Example (uncomment to run interactively):
# user_genres = get_user_preferences(all_genres)

# For demonstration purposes, we use a sample preference set here.
# Replace this line with the call above when running interactively.
user_genres = get_user_preferences(all_genres) if __name__ == "__main__" else ["Sci-Fi", "Action"]

if not user_genres:
    print("No valid genres selected — defaulting to a general sample: Sci-Fi, Action")
    user_genres = ["Sci-Fi", "Action"]

print("Your preferences:", user_genres)

# %% [markdown]
# ## Step 4: Match Preferences Using Cosine Similarity

# %%
user_vector = vectorize(user_genres, all_genres).reshape(1, -1)

similarities = cosine_similarity(user_vector, movie_vectors)[0]
df["similarity"] = similarities

# %% [markdown]
# ## Step 5: Display Recommended Items

# %%
top_n = 5
recommendations = df.sort_values("similarity", ascending=False).head(top_n)

print(f"\nTop {top_n} recommendations for you:\n")
for i, row in enumerate(recommendations.itertuples(), start=1):
    genre_str = ", ".join(row.genres)
    print(f"{i}. {row.title}  (genres: {genre_str} | match score: {row.similarity:.2f})")

# %% [markdown]
# ## Summary
# - Built a small tagged movie dataset (15 movies, 10 genres)
# - Represented movies and user preferences as multi-hot genre vectors
# - Took user input for favorite genres via the terminal
# - Used **cosine similarity** to match preferences to movies (pattern matching)
# - Displayed the top 5 recommended movies ranked by similarity score
