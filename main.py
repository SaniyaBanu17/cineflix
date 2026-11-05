import os
import pickle
import requests
import pandas as pd
import streamlit as st

# Page Configuration (Wide layout)
st.set_page_config(page_title="Cineflix - Ultimate Movie Recommendation System", page_icon="🎬", layout="wide")

# --- CUSTOM NETFLIX BLACK & RED THEME CSS ---
st.markdown("""
    <style>
    .stApp {
        background-color: #141414;
        color: #ffffff;
    }
    .main-title {
        color: #E50914;
        font-weight: 800;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #b3b3b3;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #E50914;
        color: white;
        font-weight: bold;
        border-radius: 4px;
        width: 100%;
        padding: 0.6rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #f40612;
        color: white;
    }
    .card-title {
        text-align: center;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 8px;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE FOR LOGIN ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# --- LOGIN SCREEN ---
if not st.session_state.logged_in:
    st.markdown("<h1 class='main-title'>🎬 CINEFLIX</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Unlimited movies, TV shows, trailers, and more.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("### Sign In to Cineflix")
        username = st.text_input("Username or Email")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if username and password:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.warning("Please enter your login details.")
    st.stop()

# --- BACKEND MODEL LOADING ---
SIMILARITY_URL = "https://github.com/SaniyaBanu17/cineflix/releases/download/type_v1.0/similarity.pkl"
SIMILARITY_PATH = "similarity.pkl"

if not os.path.exists(SIMILARITY_PATH) or os.path.getsize(SIMILARITY_PATH) < 1000000:
    with st.spinner("Loading Cineflix AI recommendation matrix (~176MB)... Please wait."):
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(SIMILARITY_URL, headers=headers, stream=True)
        if response.status_code == 200:
            with open(SIMILARITY_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        else:
            st.error(f"Failed to download similarity file. Status code: {response.status_code}")

@st.cache_resource
def load_data():
    with open('movie_dict.pkl', 'rb') as f:
        movie_dict = pickle.load(f)
    movies_df = pd.DataFrame(movie_dict)
    
    with open(SIMILARITY_PATH, 'rb') as f:
        similarity_matrix = pickle.load(f)
        
    return movies_df, similarity_matrix

try:
    movies, similarity = load_data()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# --- HELPER FUNCTIONS: POSTER & TMDB LINKS ---
def fetch_movie_details(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url).json()
        poster_path = data.get('poster_path')
        poster_url = "https://image.tmdb.org/t/p/w500/" + poster_path if poster_path else "https://via.placeholder.com/500x750?text=No+Image"
        homepage = data.get('homepage', f"https://www.themoviedb.org/movie/{movie_id}")
        if not homepage or homepage == "":
            homepage = f"https://www.themoviedb.org/movie/{movie_id}"
        return poster_url, homepage
    except Exception:
        return "https://via.placeholder.com/500x750?text=No+Image", f"https://www.themoviedb.org/movie/{movie_id}"

# Recommendation Logic
def recommend(movie_title):
    try:
        movie_index = movies[movies['title'].str.lower() == movie_title.lower()].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommended_movies = []
        recommended_posters = []
        recommended_links = []
        
        for i in movies_list:
            m_id = movies.iloc[i[0]].movie_id
            m_title = movies.iloc[i[0]].title
            poster, link = fetch_movie_details(m_id)
            recommended_movies.append(m_title)
            recommended_posters.append(poster)
            recommended_links.append(link)
            
        return recommended_movies, recommended_posters, recommended_links
    except Exception as e:
        st.error(f"Could not find exact match for '{movie_title}'. Try searching by keyword!")
        return [], [], []

# --- MAIN DASHBOARD INTERFACE ---
st.markdown("<h1 class='main-title'>🎬 CINEFLIX</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Explore genres, search by title, director, or actor, and get personalized recommendations.</p>", unsafe_allow_html=True)

# Sidebar filters & navigation
st.sidebar.markdown("## 🔍 Advanced Search & Filters")
search_query = st.sidebar.text_input("Search Title / Actor / Director:")
selected_genre = st.sidebar.selectbox("Filter by Genre:", ["All", "Action", "Comedy", "Drama", "Romance", "Horror", "Sci-Fi"])

# Main Selection Form
movie_list = movies['title'].values if 'title' in movies.columns else []

col1, col2 = st.columns([3, 1])
with col1:
    selected_movie_name = st.selectbox("Select a movie you love:", movie_list)

with col2:
    st.write("")
    st.write("")
    recommend_btn = st.button("Show Recommendations")

# Handling Search / Filter results or Recommendations
if search_query:
    st.markdown(f"### Search Results for: *{search_query}*")
    match_results = movies[movies['title'].str.contains(search_query, case=False, na=False)]
    if not match_results.empty:
        scols = st.columns(min(len(match_results), 5))
        for idx, row in enumerate(match_results.head(5).iterrows()):
            m_id = row[1].movie_id
            m_title = row[1].title
            poster, link = fetch_movie_details(m_id)
            with scols[idx]:
                st.image(poster, use_column_width=True)
                st.markdown(f"<p class='card-title'>{m_title}</p>", unsafe_allow_html=True)
                st.markdown(f"<a href='{link}' target='_blank'><button style='background-color:#E50914;color:white;border:none;padding:5px 10px;border-radius:4px;width:100%;cursor:pointer;'>▶ Watch / TMDB</button></a>", unsafe_allow_html=True)
    else:
        st.info("No matching titles found in search.")

elif recommend_btn or selected_movie_name:
    with st.spinner("Unlocking cinematic matches..."):
        names, posters, links = recommend(selected_movie_name)
        if names:
            st.markdown(f"### Movies Similar to *{selected_movie_name}*")
            cols = st.columns(5)
            for idx, col in enumerate(cols):
                with col:
                    st.image(posters[idx], use_column_width=True)
                    st.markdown(f"<p class='card-title'>{names[idx]}</p>", unsafe_allow_html=True)
                    st.markdown(f"<a href='{links[idx]}' target='_blank'><button style='background-color:#E50914;color:white;border:none;padding:6px 12px;border-radius:4px;width:100%;font-weight:bold;cursor:pointer;'>▶ Watch on TMDB</button></a>", unsafe_allow_html=True)