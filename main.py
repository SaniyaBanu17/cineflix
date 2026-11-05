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
        font-size: 2.5rem;
        letter-spacing: 2px;
        margin: 0;
    }
    .nav-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #141414;
        padding: 10px 0px;
        border-bottom: 1px solid #282828;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #E50914;
        color: white;
        font-weight: bold;
        border-radius: 4px;
        width: 100%;
        padding: 0.5rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #f40612;
        color: white;
    }
    .hero-box {
        background: linear-gradient(180deg, rgba(20,20,20,0.2) 0%, rgba(20,20,20,1) 100%), url('https://image.tmdb.org/t/p/original/rAiYTfKGqDCRIIqo6LEuPJ19zB.jpg');
        background-size: cover;
        background-position: center;
        padding: 50px;
        border-radius: 8px;
        margin-bottom: 30px;
    }
    .card-title {
        text-align: center;
        font-weight: 700;
        font-size: 0.95rem;
        margin-top: 8px;
        color: #ffffff;
    }
    .card-info {
        text-align: center;
        color: #b3b3b3;
        font-size: 0.8rem;
        margin-bottom: 5px;
    }
    .movie-desc {
        background-color: #1f1f1f;
        padding: 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        color: #cccccc;
        margin-bottom: 8px;
        min-height: 70px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"
if "selected_genre" not in st.session_state:
    st.session_state.selected_genre = "All"
if "show_info" not in st.session_state:
    st.session_state.show_info = False
if "show_trailer" not in st.session_state:
    st.session_state.show_trailer = False

# --- LOGIN SCREEN ---
if not st.session_state.logged_in:
    st.markdown("<h1 style='color: #E50914; text-align: center;'>🎬 CF CINEFLIX</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("### Sign In")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username and password:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.warning("Please enter your credentials.")
    st.stop()

# --- BACKEND MODEL LOADING ---
SIMILARITY_URL = "https://github.com/SaniyaBanu17/cineflix/releases/download/type_v1.0/similarity.pkl"
SIMILARITY_PATH = "similarity.pkl"

if not os.path.exists(SIMILARITY_PATH) or os.path.getsize(SIMILARITY_PATH) < 1000000:
    with st.spinner("Loading Cineflix recommendation matrices (~176MB)..."):
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(SIMILARITY_URL, headers=headers, stream=True)
        if response.status_code == 200:
            with open(SIMILARITY_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

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
    st.error(f"Error loading files: {e}")
    st.stop()

# --- HELPER FUNCTIONS ---
def fetch_movie_details(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url).json()
        poster_path = data.get('poster_path')
        poster_url = "https://image.tmdb.org/t/p/w500/" + poster_path if poster_path else "https://via.placeholder.com/500x750?text=No+Image"
        overview = data.get('overview', 'No description available.')
        release_date = data.get('release_date', '2014')
        year = release_date[:4] if release_date else '2014'
        rating = data.get('vote_average', 8.6)
        homepage = data.get('homepage') or f"https://www.themoviedb.org/movie/{movie_id}"
        return poster_url, overview, year, rating, homepage
    except Exception:
        return "https://via.placeholder.com/500x750?text=No+Image", "Description unavailable.", "2014", 8.5, f"https://www.themoviedb.org/movie/{movie_id}"

def recommend(movie_title):
    try:
        match_idx = movies[movies['title'].str.lower() == movie_title.lower()].index
        if len(match_idx) == 0:
            match_idx = movies[movies['title'].str.contains(movie_title, case=False, na=False)].index
        if len(match_idx) == 0:
            return [], [], [], [], [], []
        
        movie_index = match_idx[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        titles, posters, overviews, metadata, links, ids = [], [], [], [], [], []
        for i in movies_list:
            m_id = movies.iloc[i[0]].movie_id
            m_title = movies.iloc[i[0]].title
            poster, overview, year, rating, link = fetch_movie_details(m_id)
            titles.append(m_title)
            posters.append(poster)
            overviews.append(overview)
            metadata.append(f"{year} | ⭐ {rating}/10")
            links.append(link)
            ids.append(m_id)
        return titles, posters, overviews, metadata, links, ids
    except Exception:
        return [], [], [], [], [], []

# --- HORIZONTAL NETFLIX NAVIGATION BAR ---
col_logo, col_home, col_movies, col_new, col_watchlist, col_search = st.columns([1.2, 0.8, 0.9, 1.2, 1, 1.8])

with col_logo:
    st.markdown("<h2 class='main-title'>🎬 CF</h2>", unsafe_allow_html=True)
with col_home:
    if st.button("Home"):
        st.session_state.current_page = "Home"
        st.session_state.selected_genre = "All"
with col_movies:
    if st.button("Movies"):
        st.session_state.current_page = "Movies"
with col_new:
    if st.button("New & Popular"):
        st.session_state.current_page = "New & Popular"
with col_watchlist:
    if st.button(f"Watchlist ({len(st.session_state.watchlist)})"):
        st.session_state.current_page = "Watchlist"
with col_search:
    search_query = st.text_input("", placeholder="Search movies, actors, directors...", label_visibility="collapsed")

st.markdown("---")

# --- RENDER PAGES ACCORDING TO NAVIGATION ---

# 1. WATCHLIST PAGE
if st.session_state.current_page == "Watchlist":
    st.markdown("## 📌 My Watchlist")
    if len(st.session_state.watchlist) == 0:
        st.info("Your watchlist is empty. Add movies from recommendations!")
    else:
        w_cols = st.columns(min(len(st.session_state.watchlist), 5))
        for idx, item in enumerate(st.session_state.watchlist):
            with w_cols[idx % 5]:
                st.image(item['poster'], use_column_width=True)
                st.markdown(f"<p class='card-title'>{item['title']}</p>", unsafe_allow_html=True)
                if st.button("❌ Remove", key=f"rem_{idx}"):
                    st.session_state.watchlist.pop(idx)
                    st.rerun()

# 2. MOVIES / GENRES PAGE
elif st.session_state.current_page in ["Movies", "New & Popular"] or st.session_state.selected_genre != "All":
    st.markdown("## 🎭 Explore Categories & Genres")
    
    # Genre Selection Horizontal Buttons
    g_col1, g_col2, g_col3, g_col4, g_col5, g_col6, g_col7 = st.columns(7)
    with g_col1:
        if st.button("All"): st.session_state.selected_genre = "All"
    with g_col2:
        if st.button("Action"): st.session_state.selected_genre = "Action"
    with g_col3:
        if st.button("Comedy"): st.session_state.selected_genre = "Comedy"
    with g_col4:
        if st.button("Drama"): st.session_state.selected_genre = "Drama"
    with g_col5:
        if st.button("Horror"): st.session_state.selected_genre = "Horror"
    with g_col6:
        if st.button("Sci-Fi"): st.session_state.selected_genre = "Sci-Fi"
    with g_col7:
        if st.button("Romance"): st.session_state.selected_genre = "Romance"
        
    st.markdown(f"### Showing Category: {st.session_state.selected_genre}")
    
    cat_movies = movies.sample(n=min(10, len(movies)), random_state=42)
    c_cols = st.columns(5)
    for idx, row in enumerate(cat_movies.iterrows()):
        m_id = row[1].movie_id
        m_title = row[1].title
        poster, overview, year, rating, link = fetch_movie_details(m_id)
        with c_cols[idx % 5]:
            st.image(poster, use_column_width=True)
            st.markdown(f"<p class='card-title'>{m_title}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='card-info'>{year} | ⭐ {rating}/10</p>", unsafe_allow_html=True)
            if st.button("➕ Watchlist", key=f"cat_add_{idx}"):
                st.session_state.watchlist.append({"title": m_title, "poster": poster, "link": link})
                st.success("Added!")

# 3. SEARCH RESULTS (IF SEARCH QUERY ENTERED)
elif search_query:
    st.markdown(f"### 🔍 Search Results for: *{search_query}*")
    mask = movies['title'].str.contains(search_query, case=False, na=False)
    matched_df = movies[mask].head(5)
    if not matched_df.empty:
        s_cols = st.columns(min(len(matched_df), 5))
        for idx, row in enumerate(matched_df.iterrows()):
            m_id = row[1].movie_id
            m_title = row[1].title
            poster, overview, year, rating, link = fetch_movie_details(m_id)
            with s_cols[idx]:
                st.image(poster, use_column_width=True)
                st.markdown(f"<p class='card-title'>{m_title}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='card-info'>{year} | ⭐ {rating}/10</p>", unsafe_allow_html=True)
                st.markdown(f"<a href='{link}' target='_blank'><button style='background-color:#E50914;color:white;border:none;padding:5px;border-radius:4px;width:100%;'>▶ Watch on TMDB</button></a>", unsafe_allow_html=True)
    else:
        st.info("No matching movies found in database.")

# 4. HOME PAGE (DEFAULT)
else:
    # HERO BANNER: INTERSTELLAR
    st.markdown("""
        <div class='hero-box'>
            <h1 style='font-size: 3.2rem; color: white; margin-bottom: 5px;'>INTERSTELLAR</h1>
            <p style='color: #46d369; font-weight: bold; font-size: 1.1rem;'>2014 | ⭐ 8.6/10 | Sci-Fi, Adventure, Drama</p>
            <p style='max-width: 650px; color: #ddd; font-size: 1rem;'>
                When Earth becomes uninhabitable in the future, a farmer and ex-NASA pilot, Joseph Cooper, is tasked to pilot a spacecraft, along with a team of researchers, to find a new planet for humans.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Hero Action Buttons (Play Trailer & Info)
    h_col1, h_col2, h_col3 = st.columns([1, 1, 3])
    with h_col1:
        if st.button("▶ Play Trailer"):
            st.session_state.show_trailer = not st.session_state.show_trailer
    with h_col2:
        if st.button("ℹ️ Info"):
            st.session_state.show_info = not st.session_state.show_info

    # Toggle Info or Trailer display for Interstellar
    if st.session_state.show_trailer:
        st.video("https://www.youtube.com/watch?v=zSWdZVtXT7E")
    if st.session_state.show_info:
        st.info("Interstellar (2014) directed by Christopher Nolan. Cast: Matthew McConaughey, Anne Hathaway, Jessica Chastain. Epic space-time exploration masterpiece.")

    st.markdown("<br>", unsafe_allow_html=True)

    # MAIN RECOMMENDATION INPUT ROW
    movie_list = movies['title'].values if 'title' in movies.columns else []
    r_col1, r_col2 = st.columns([3, 1])
    with r_col1:
        selected_movie = st.selectbox("Select or search a movie for recommendations:", movie_list)
    with r_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        rec_btn = st.button("Show Recommendations")

    # RECOMMENDATIONS GRID DISPLAY
    if rec_btn or selected_movie:
        with st.spinner("Fetching recommended movies based on your activity..."):
            names, posters, overviews, metadata, links, ids = recommend(selected_movie)
            if names:
                st.markdown(f"### Recommended for You (Based on *{selected_movie}*)")
                cols = st.columns(5)
                for idx, col in enumerate(cols):
                    with col:
                        st.image(posters[idx], use_column_width=True)
                        st.markdown(f"<p class='card-title'>{names[idx]}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p class='card-info'>{metadata[idx]}</p>", unsafe_allow_html=True)
                        st.markdown(f"<div class='movie-desc'>{overviews[idx]}</div>", unsafe_allow_html=True)
                        
                        # Add to Watchlist
                        if st.button("➕ Add to Watchlist", key=f"home_add_{idx}"):
                            st.session_state.watchlist.append({"title": names[idx], "poster": posters[idx], "link": links[idx]})
                            st.success("Added!")
                        
                        # Watch on TMDB
                        st.markdown(f"<a href='{links[idx]}' target='_blank'><button style='background-color:#E50914;color:white;border:none;padding:6px;border-radius:4px;width:100%;font-weight:bold;margin-top:5px;'>▶ Watch on TMDB</button></a>", unsafe_allow_html=True)