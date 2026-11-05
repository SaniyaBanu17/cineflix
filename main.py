import os
import pickle
import requests
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(page_title="Cineflix - Watch & Discover", page_icon="🎬", layout="wide")

# --- CUSTOM NETFLIX TYPOGRAPHY & THEME CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@400;600;700;800&display=swap');

    .stApp {
        background-color: #141414;
        color: #ffffff;
        font-family: 'Montserrat', sans-serif;
    }
    
    .main-title {
        color: #E50914;
        font-family: 'Bebas Neue', cursive;
        font-weight: 800;
        font-size: 2.8rem;
        letter-spacing: 2px;
        margin: 0;
        padding-top: 5px;
    }
    
    .section-header {
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        font-size: 1.4rem;
        margin-top: 20px;
        margin-bottom: 15px;
        color: #ffffff;
    }

    .hero-box {
        background: linear-gradient(180deg, rgba(20,20,20,0.3) 0%, rgba(20,20,20,1) 100%), url('https://image.tmdb.org/t/p/original/rAiYTfKGqDCRIIqo6LEuPJ19zB.jpg');
        background-size: cover;
        background-position: center;
        padding: 50px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.8);
    }
    
    .hero-title {
        font-family: 'Bebas Neue', cursive;
        font-size: 4rem;
        color: white;
        margin-top: 5px;
        margin-bottom: 5px;
        letter-spacing: 2px;
    }

    .stButton>button {
        background-color: #E50914;
        color: white;
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        border-radius: 4px;
        width: 100%;
        padding: 0.5rem;
        border: none;
        transition: all 0.2s ease-in-out;
    }

    .stButton>button:hover {
        background-color: #f40612;
        color: white;
        transform: scale(1.02);
    }

    .card-title {
        font-family: 'Montserrat', sans-serif;
        text-align: center;
        font-weight: 700;
        font-size: 0.95rem;
        margin-top: 8px;
        margin-bottom: 2px;
        color: #ffffff;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .card-info {
        font-family: 'Montserrat', sans-serif;
        text-align: center;
        color: #b3b3b3;
        font-size: 0.8rem;
        margin-bottom: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"
if "selected_genre" not in st.session_state:
    st.session_state.selected_genre = "All"
if "show_details" not in st.session_state:
    st.session_state.show_details = False

# --- BACKEND MODEL LOADING ---
SIMILARITY_URL = "https://github.com/SaniyaBanu17/cineflix/releases/download/type_v1.0/similarity.pkl"
SIMILARITY_PATH = "similarity.pkl"

if not os.path.exists(SIMILARITY_PATH) or os.path.getsize(SIMILARITY_PATH) < 1000000:
    with st.spinner("Loading recommendation matrices (~176MB)..."):
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

def search_all_columns(df, query):
    if not query:
        return df.head(0)
    query_str = str(query).lower().strip()
    title_mask = df['title'].astype(str).str.lower().str.contains(query_str, na=False)
    results = df[title_mask]
    
    if len(results) < 5:
        other_masks = []
        for col in df.columns:
            if col != 'similarity':
                other_masks.append(df[col].astype(str).str.lower().str.contains(query_str, na=False))
        combined_mask = pd.concat(other_masks, axis=1).any(axis=1)
        results = df[combined_mask]
        
    return results

# --- NETFLIX HORIZONTAL NAVBAR ---
nav1, nav2, nav3, nav4, nav5, nav6, nav7 = st.columns([1.5, 0.7, 0.7, 1.1, 0.9, 1.1, 1.8])

with nav1:
    st.markdown("<h1 class='main-title'>CINEFLIX</h1>", unsafe_allow_html=True)
with nav2:
    if st.button("Home"): st.session_state.current_page = "Home"
with nav3:
    if st.button("Movies"): st.session_state.current_page = "Movies"
with nav4:
    if st.button("New & Popular"): st.session_state.current_page = "New & Popular"
with nav5:
    if st.button(f"Watchlist ({len(st.session_state.watchlist)})"): st.session_state.current_page = "Watchlist"
with nav6:
    field_filter = st.selectbox("", ["All Fields", "Title", "Actor", "Director"], label_visibility="collapsed")
with nav7:
    search_query = st.text_input("", placeholder="Search movies, actors, directors...", label_visibility="collapsed")

st.markdown("<hr style='border-color: #282828;'>", unsafe_allow_html=True)

# --- ROUTING PAGES ---

if st.session_state.current_page == "Watchlist":
    st.markdown("<h2 class='section-header'>📌 My Watchlist</h2>", unsafe_allow_html=True)
    if len(st.session_state.watchlist) == 0:
        st.info("Your watchlist is empty.")
    else:
        w_cols = st.columns(min(len(st.session_state.watchlist), 5))
        for idx, item in enumerate(st.session_state.watchlist):
            with w_cols[idx % 5]:
                st.image(item['poster'], use_column_width=True)
                st.markdown(f"<p class='card-title'>{item['title']}</p>", unsafe_allow_html=True)
                if st.button("❌ Remove", key=f"rem_{idx}"):
                    st.session_state.watchlist.pop(idx)
                    st.rerun()

elif search_query:
    st.markdown(f"<h3 class='section-header'>Search Results for: <i>{search_query}</i></h3>", unsafe_allow_html=True)
    matched_df = search_all_columns(movies, search_query).head(8)
    
    if not matched_df.empty:
        s_cols = st.columns(min(len(matched_df), 4))
        for idx, row in enumerate(matched_df.iterrows()):
            m_id = row[1].movie_id if 'movie_id' in row[1] else row[1].get('id', 0)
            m_title = row[1].title
            poster, overview, year, rating, link = fetch_movie_details(m_id)
            with s_cols[idx % 4]:
                st.image(poster, use_column_width=True)
                st.markdown(f"<p class='card-title'>{m_title}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='card-info'>{year} | ⭐ {rating}/10</p>", unsafe_allow_html=True)
                if st.button("➕ Watchlist", key=f"search_add_{idx}"):
                    st.session_state.watchlist.append({"title": m_title, "poster": poster, "link": link})
                    st.success("Added!")
                st.markdown(f"<a href='{link}' target='_blank'><button style='background-color:#E50914;color:white;font-family:Montserrat,sans-serif;font-weight:700;border:none;padding:6px;border-radius:4px;width:100%;margin-top:4px;cursor:pointer;'>▶ Watch on TMDB</button></a>", unsafe_allow_html=True)
    else:
        st.info(f"No matching movies, directors, or actors found for '{search_query}'.")

else:
    # --- GENRES HORIZONTAL BAR ---
    st.markdown("<p style='font-weight:700; font-size:1.1rem;'>GENRES: &nbsp;&nbsp;&nbsp; " + " &nbsp;&nbsp;&nbsp; ".join([f"{g}" for g in ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Romance"]]) + "</p>", unsafe_allow_html=True)
    
    genre_col1, genre_col2, genre_col3, genre_col4, genre_col5, genre_col6 = st.columns(6)
    with genre_col1: 
        if st.button("Action"): st.session_state.selected_genre = "Action"
    with genre_col2: 
        if st.button("Comedy"): st.session_state.selected_genre = "Comedy"
    with genre_col3: 
        if st.button("Drama"): st.session_state.selected_genre = "Drama"
    with genre_col4: 
        if st.button("Horror"): st.session_state.selected_genre = "Horror"
    with genre_col5: 
        if st.button("Sci-Fi"): st.session_state.selected_genre = "Sci-Fi"
    with genre_col6: 
        if st.button("Romance"): st.session_state.selected_genre = "Romance"

    st.markdown("<br>", unsafe_allow_html=True)

    # --- HERO BANNER: INTERSTELLAR ---
    st.markdown("""
        <div class='hero-box'>
            <p style='color: #E50914; font-weight: 800; font-size: 0.9rem; margin-bottom: 0px; letter-spacing: 1px;'>CF SERIES • FEATURED</p>
            <div class='hero-title'>Interstellar</div>
            <p style='max-width: 620px; color: #dddddd; font-size: 0.95rem; line-height: 1.5;'>
                When Earth becomes uninhabitable, a team of ex-NASA pilots undertakes the most important mission in human history: traveling beyond this galaxy to discover whether mankind has a future among the stars.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Play Trailer & Info Button
    if st.button("▶ Play Trailer & Info"):
        st.session_state.show_details = not st.session_state.show_details

    if st.session_state.show_details:
        st.info("⭐ Interstellar (2014) | Rating: 8.6/10 | Director: Christopher Nolan\n\nEpic space adventure exploring gravity, time dilation, and survival.")
        st.video("https://www.youtube.com/watch?v=zSWdZVtXT7E")
        st.markdown(f"<a href='https://www.themoviedb.org/movie/157336' target='_blank'><button style='background-color:#E50914;color:white;font-family:Montserrat,sans-serif;font-weight:700;border:none;padding:8px;border-radius:4px;width:200px;cursor:pointer;'>Watch on TMDB</button></a>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- RECOMMENDED FOR ADMIN SECTION ---
    st.markdown("<h3 class='section-header'>Recommended for <span style='color:#E50914;'>admin</span> <span style='font-size:0.8rem; color:#b3b3b3; font-weight:400;'>(Based on your activity)</span></h3>", unsafe_allow_html=True)
    
    rec_sample = movies.sample(n=4, random_state=101)
    r_cols = st.columns(4)
    for idx, row in enumerate(rec_sample.iterrows()):
        m_id = row[1].movie_id if 'movie_id' in row[1] else row[1].get('id', 0)
        m_title = row[1].title
        poster, overview, year, rating, link = fetch_movie_details(m_id)
        with r_cols[idx]:
            st.image(poster, use_column_width=True)
            st.markdown(f"<p class='card-title'>{m_title}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='card-info'>{year} | ⭐ {rating}/10</p>", unsafe_allow_html=True)
            if st.button("➕ Watchlist", key=f"rec_{idx}"):
                st.session_state.watchlist.append({"title": m_title, "poster": poster, "link": link})
                st.success("Added!")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- TRENDING & RECOMMENDED MOVIES SECTION ---
    st.markdown("<h3 class='section-header'>Trending & Recommended Movies</h3>", unsafe_allow_html=True)
    trend_sample = movies.sample(n=8, random_state=202)
    t_cols = st.columns(4)
    for idx, row in enumerate(trend_sample.iterrows()):
        m_id = row[1].movie_id if 'movie_id' in row[1] else row[1].get('id', 0)
        m_title = row[1].title
        poster, overview, year, rating, link = fetch_movie_details(m_id)
        with t_cols[idx % 4]:
            st.image(poster, use_column_width=True)
            st.markdown(f"<p class='card-title'>{m_title}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='card-info'>{year} | ⭐ {rating}/10</p>", unsafe_allow_html=True)
            st.markdown(f"<a href='{link}' target='_blank'><button style='background-color:#E50914;color:white;font-family:Montserrat,sans-serif;font-weight:700;border:none;padding:5px;border-radius:4px;width:100%;margin-top:4px;cursor:pointer;'>▶ TMDB / Watch</button></a>", unsafe_allow_html=True)