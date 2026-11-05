import os
import pickle
import requests
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(page_title="Cineflix - Watch & Discover", page_icon="🎬", layout="wide")

# --- MATCHING EXACT STYLING FROM REFERENCE IMAGE ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #141414;
        color: #ffffff;
        font-family: 'Montserrat', sans-serif;
    }
    
    .main-title {
        color: #E50914;
        font-family: 'Bebas Neue', cursive;
        font-size: 2.8rem;
        letter-spacing: 1px;
        margin: 0;
    }
    
    .section-header {
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        font-size: 1.3rem;
        margin-top: 25px;
        margin-bottom: 15px;
        color: #ffffff;
    }

    /* Exact Interstellar Hero Banner Layout */
    .hero-box {
        background: linear-gradient(180deg, rgba(20,20,20,0.1) 0%, rgba(20,20,20,0.85) 100%), url('https://image.tmdb.org/t/p/original/rAiYTfKGqDCRIIqo6LEuPJ19zB.jpg');
        background-size: cover;
        background-position: center top;
        padding: 70px 45px;
        border-radius: 8px;
        margin-bottom: 15px;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.9);
    }

    /* Exact white button styling with dark text matching screenshot */
    .stButton>button {
        background-color: #f0f0f0;
        color: #141414;
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.5rem 1.2rem;
        border: none;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    }

    .stButton>button:hover {
        background-color: #ffffff;
        color: #000000;
    }

    .card-title {
        text-align: center;
        font-weight: 600;
        font-size: 0.9rem;
        margin-top: 6px;
        color: #ffffff;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .card-info {
        text-align: center;
        color: #b3b3b3;
        font-size: 0.75rem;
        margin-bottom: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "users_db" not in st.session_state:
    st.session_state.users_db = {"admin": "password123", "saniya": "San123"}
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"
if "selected_genre" not in st.session_state:
    st.session_state.selected_genre = "All"
if "show_details" not in st.session_state:
    st.session_state.show_details = False

# --- AUTHENTICATION SCREEN ---
if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 class='main-title' style='text-align:center;'>🎬 CINEFLIX</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #b3b3b3; margin-bottom: 30px;'>Unlimited movies, TV shows, and recommendations.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        tab1, tab2, tab3 = st.tabs(["Sign In", "Register", "Forgot Password"])
        
        with tab1:
            st.markdown("### Sign In")
            l_user = st.text_input("Username", key="l_user")
            l_pass = st.text_input("Password", type="password", key="l_pass")
            if st.button("Login"):
                if l_user in st.session_state.users_db and st.session_state.users_db[l_user] == l_pass:
                    st.session_state.logged_in = True
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
                    
        with tab2:
            st.markdown("### Create Account")
            r_user = st.text_input("Choose Username", key="r_user")
            r_pass = st.text_input("Choose Password", type="password", key="r_pass")
            if st.button("Sign Up"):
                if r_user and r_pass:
                    if r_user in st.session_state.users_db:
                        st.warning("Username already exists.")
                    else:
                        st.session_state.users_db[r_user] = r_pass
                        st.success("Account created successfully! Please log in.")
                else:
                    st.warning("Please fill in all fields.")
                    
        with tab3:
            st.markdown("### Reset Password")
            f_user = st.text_input("Username", key="f_user")
            f_pass = st.text_input("New Password", type="password", key="f_pass")
            if st.button("Update Password"):
                if f_user in st.session_state.users_db:
                    st.session_state.users_db[f_user] = f_pass
                    st.success("Password updated successfully! You can now log in.")
                else:
                    st.error("Username not found.")
    st.stop()

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
        other_masks = [df[col].astype(str).str.lower().str.contains(query_str, na=False) for col in df.columns if col != 'similarity']
        combined_mask = pd.concat(other_masks, axis=1).any(axis=1)
        results = df[combined_mask]
    return results

# --- NETFLIX HORIZONTAL NAVBAR ---
nav1, nav2, nav3, nav4, nav5, nav6, nav7 = st.columns([1.5, 0.7, 0.8, 1.1, 1, 1.1, 1.8])

with nav1:
    st.markdown("<h1 class='main-title'>CINEFLIX</h1>", unsafe_allow_html=True)
with nav2:
    if st.button("Home"): st.session_state.current_page = "Home"; st.session_state.selected_genre = "All"
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

st.markdown("<hr style='border-color: #282828; margin-top: 5px; margin-bottom: 15px;'>", unsafe_allow_html=True)

# --- PAGES ROUTING ---

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
        s_cols = st.columns(4)
        for idx, row in enumerate(matched_df.iterrows()):
            m_id = row[1].movie_id if 'movie_id' in row[1] else row[1].get('id', 0)
            m_title = row[1].title
            poster, overview, year, rating, link = fetch_movie_details(m_id)
            with s_cols[idx % 4]:
                st.image(poster, use_column_width=True)
                st.markdown(f"<p class='card-title'>{m_title}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='card-info'>{year} | ⭐ {rating}/10</p>", unsafe_allow_html=True)
                if st.button("➕ Watchlist", key=f"s_add_{idx}"):
                    st.session_state.watchlist.append({"title": m_title, "poster": poster, "link": link})
                    st.success("Added!")
    else:
        st.info(f"No results found for '{search_query}'.")

elif st.session_state.current_page in ["Movies", "New & Popular"] or st.session_state.selected_genre != "All":
    st.markdown(f"<h2 class='section-header'>🎬 {st.session_state.current_page} - {st.session_state.selected_genre} Movies</h2>", unsafe_allow_html=True)
    
    display_df = search_all_columns(movies, st.session_state.selected_genre) if st.session_state.selected_genre != "All" else movies.sample(n=12, random_state=42)
    if display_df.empty:
        display_df = movies.sample(n=8, random_state=12)

    m_cols = st.columns(4)
    for idx, row in enumerate(display_df.head(12).iterrows()):
        m_id = row[1].movie_id if 'movie_id' in row[1] else row[1].get('id', 0)
        m_title = row[1].title
        poster, overview, year, rating, link = fetch_movie_details(m_id)
        with m_cols[idx % 4]:
            st.image(poster, use_column_width=True)
            st.markdown(f"<p class='card-title'>{m_title}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='card-info'>{year} | ⭐ {rating}/10</p>", unsafe_allow_html=True)
            if st.button("➕ Watchlist", key=f"cat_add_{idx}"):
                st.session_state.watchlist.append({"title": m_title, "poster": poster, "link": link})
                st.success("Added!")

else:
    # --- CLEAN GENRES NAVIGATION ---
    st.markdown("""
        <div style="display: flex; gap: 15px; font-weight: 600; font-size: 0.95rem; margin-bottom: 15px; color: #b3b3b3; align-items: center;">
            <span style="color: #ffffff; font-weight: 700;">GENRES:</span>
        </div>
    """, unsafe_allow_html=True)

    gc1, gc2, gc3, gc4, gc5, gc6 = st.columns(6)
    with gc1:
        if st.button("Action"): st.session_state.selected_genre = "Action"; st.rerun()
    with gc2:
        if st.button("Comedy"): st.session_state.selected_genre = "Comedy"; st.rerun()
    with gc3:
        if st.button("Drama"): st.session_state.selected_genre = "Drama"; st.rerun()
    with gc4:
        if st.button("Horror"): st.session_state.selected_genre = "Horror"; st.rerun()
    with gc5:
        if st.button("Sci-Fi"): st.session_state.selected_genre = "Sci-Fi"; st.rerun()
    with gc6:
        if st.button("Romance"): st.session_state.selected_genre = "Romance"; st.rerun()

    # --- EXACT INTERSTELLAR HERO BANNER MATCHING SCREENSHOT ---
    st.markdown("""
        <div class='hero-box'>
            <span style='background-color: #E50914; color: white; padding: 3px 6px; font-size: 0.75rem; font-weight: 700; border-radius: 3px;'>CF</span>
            <span style='color: #ffffff; font-weight: 600; font-size: 0.85rem; letter-spacing: 1px; margin-left: 6px;'>SERIES • FEATURED</span>
            <h1 style='font-size: 3.5rem; color: white; margin-top: 5px; margin-bottom: 8px; font-family: "Bebas Neue", cursive; letter-spacing: 2px;'>Interstellar</h1>
            <p style='max-width: 580px; color: #dddddd; font-size: 0.95rem; line-height: 1.4;'>
                When Earth becomes uninhabitable, a team of ex-NASA pilots undertakes the most important mission in human history: traveling beyond this galaxy to discover whether mankind has a future among the stars.
            </p>
        </div>
    """, unsafe_allow_html=True)

    if st.button("▶ Play Trailer & Info"):
        st.session_state.show_details = not st.session_state.show_details

    if st.session_state.show_details:
        st.info("⭐ Interstellar (2014) | Rating: 8.6/10 | Director: Christopher Nolan\n\nEpic space adventure exploring gravity, time dilation, and survival.")
        st.video("https://www.youtube.com/watch?v=zSWdZVtXT7E")
        st.markdown(f"<a href='https://www.themoviedb.org/movie/157336' target='_blank'><button style='background-color:#ffffff;color:#141414;border-radius:6px;padding:8px 16px;border:none;font-weight:600;'>Watch on TMDB</button></a>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- HOMEPAGE GRIDS ---
    st.markdown("<h3 class='section-header'>Recommended for user <span style='font-weight:400; font-size:0.8rem; color:#b3b3b3;'>(Based on your activity)</span></h3>", unsafe_allow_html=True)
    rec_sample = movies.sample(n=4, random_state=77)
    r_cols = st.columns(4)
    for idx, row in enumerate(rec_sample.iterrows()):
        m_id = row[1].movie_id if 'movie_id' in row[1] else row[1].get('id', 0)
        m_title = row[1].title
        poster, overview, year, rating, link = fetch_movie_details(m_id)
        with r_cols[idx]:
            st.image(poster, use_column_width=True)
            st.markdown(f"<p class='card-title'>{m_title}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='card-info'>{year} | ⭐ {rating}/10</p>", unsafe_allow_html=True)
            if st.button("➕ Watchlist", key=f"home_rec_{idx}"):
                st.session_state.watchlist.append({"title": m_title, "poster": poster, "link": link})
                st.success("Added!")

    st.markdown("<h3 class='section-header'>Trending & Recommended Movies</h3>", unsafe_allow_html=True)
    trend_sample = movies.sample(n=8, random_state=120)
    t_cols = st.columns(4)
    for idx, row in enumerate(trend_sample.iterrows()):
        m_id = row[1].movie_id if 'movie_id' in row[1] else row[1].get('id', 0)
        m_title = row[1].title
        poster, overview, year, rating, link = fetch_movie_details(m_id)
        with t_cols[idx % 4]:
            st.image(poster, use_column_width=True)
            st.markdown(f"<p class='card-title'>{m_title}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='card-info'>{year} | ⭐ {rating}/10</p>", unsafe_allow_html=True)
            if st.button("➕ Watchlist", key=f"home_trend_{idx}"):
                st.session_state.watchlist.append({"title": m_title, "poster": poster, "link": link})
                st.success("Added!")