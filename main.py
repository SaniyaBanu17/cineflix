import os
import pickle
import requests
import pandas as pd
import streamlit as st

# GitHub Release download link for the large 176MB similarity.pkl file
SIMILARITY_URL = "https://github.com/SaniyaBanu17/cineflix/releases/download/v1.0/similarity.pkl"
SIMILARITY_PATH = "similarity.pkl"

# Auto-download similarity.pkl if it doesn't exist locally (for Streamlit Cloud)
if not os.path.exists(SIMILARITY_PATH):
    with st.spinner("Downloading model files (~176MB)... Please wait a moment. This only happens on initial startup."):
        response = requests.get(SIMILARITY_URL)
        with open(SIMILARITY_PATH, "wb") as f:
            f.write(response.content)

# Page Configuration
st.set_page_config(page_title="Cineflix - Movie Recommendation System", page_icon="🎬", layout="wide")

st.title("🎬 Cineflix Movie Recommendation System")
st.write("Select a movie you like, and we'll recommend similar titles with posters for you!")

# Function to fetch movie posters from TMDB API
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
    except Exception:
        pass
    return "https://via.placeholder.com/500x750?text=No+Image"

# Load data safely using st.cache_resource
@st.cache_resource
def load_data():
    movie_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies_df = pd.DataFrame(movie_dict)
    similarity_matrix = pickle.load(open(SIMILARITY_PATH, 'rb'))
    return movies_df, similarity_matrix

try:
    movies, similarity = load_data()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Recommendation logic
def recommend(movie):
    try:
        movie_index = movies[movies['title'] == movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommended_movies = []
        recommended_movies_posters = []
        
        for i in movies_list:
            movie_id = movies.iloc[i[0]].movie_id
            recommended_movies.append(movies.iloc[i[0]].title)
            recommended_movies_posters.append(fetch_poster(movie_id))
            
        return recommended_movies, recommended_movies_posters
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return [], []

# User Interface
movie_list = movies['title'].values if 'title' in movies.columns else ["Avatar", "The Dark Knight", "Inception"]
selected_movie_name = st.selectbox("Choose a movie:", movie_list)

if st.button("Show Recommendations"):
    names, posters = recommend(selected_movie_name)
    if names:
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            with col:
                st.text(names[idx])
                st.image(posters[idx], use_column_width=True)