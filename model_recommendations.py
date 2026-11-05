import pandas as pd
import ast

# 1. Load datasets
movies = pd.read_csv('tmdb.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# 2. Merge datasets
movies = movies.merge(credits, on='title')

# 3. Keep only the useful columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Drop rows with missing values
movies.dropna(inplace=True)

# 4. Helper function to extract names from JSON-like strings
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

# Apply helper to genres and keywords
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Helper function for cast (get top 3 actors)
def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert3)

# Helper function to get the director's name from crew
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# Clean up spaces in names (e.g., "Sam Worthington" -> "SamWorthington")
def collapse(L):
    return [i.replace(" ", "") for i in L]

movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)

# Convert overview from string to list of words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# 5. Create a single 'tags' column combining everything
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create a final dataframe with just movie_id, title, and tags
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

print("Preprocessing complete! Final dataframe shape:", new_df.shape)
print(new_df.head(1))
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Vectorize the tags using CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# 2. Calculate the cosine similarity matrix
similarity = cosine_similarity(vectors)

# 3. Define the recommendation function
def recommend(movie):
    # Find the index of the movie in the dataframe
    movie_index = new_df[new_df['title'] == movie].index[0]
    
    # Get the similarity scores for that movie with all other movies
    distances = similarity[movie_index]
    
    # Sort the movies based on similarity score (highest first), keeping top 5 (excluding itself)
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    # Print the recommended movie titles
    print(f"\nRecommendations for '{movie}':")
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

# Test the recommendation function!
recommend('Avatar')
import pickle

# Save the dataframe and similarity matrix as pickle files
pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("Model and dataframe successfully saved as pickle files!")