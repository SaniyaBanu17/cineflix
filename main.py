import os
import ast
import pickle
import re
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from functools import lru_cache
import os
import requests
import streamlit as st

SIMILARITY_URL = "https://github.com/SaniyaBanu17/cineflix/releases/download/type_v1.0/similarity.pkl"
SIMILARITY_PATH = "similarity.pkl"

if not os.path.exists(SIMILARITY_PATH):
  with st.spinner(
      "Downloading model files (~176MB)... Please wait a moment."
  ):
    response = requests.get(SIMILARITY_URL)
    with open(SIMILARITY_PATH, "wb") as f:
      f.write(response.content)

app = FastAPI(title="Cineflix Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset and similarity matrix
try:
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    new_df = pd.DataFrame(movies_dict)
except Exception as e:
    print(f"Error loading movie_dict.pkl: {e}")
    new_df = pd.DataFrame()

try:
    similarity = pickle.load(open('similarity.pkl', 'rb'))
except Exception as e:
    similarity = []

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
@lru_cache(maxsize=2000)
def fetch_poster(movie_id, row_poster=None):
    if row_poster and isinstance(row_poster, str) and row_poster.startswith('http'):
        return row_poster
    if row_poster and isinstance(row_poster, str) and row_poster.startswith('/'):
        return f"https://image.tmdb.org/t/p/w500{row_poster}"
        
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url, timeout=1)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        pass
    return "https://via.placeholder.com/500x750?text=No+Image"

def parse_metadata_to_text(val):
    if pd.isna(val) or val is None:
        return ""
    
    if isinstance(val, list):
        items = []
        for item in val:
            if isinstance(item, dict):
                for key in ['name', 'title', 'character', 'job', 'director', 'original_name']:
                    if key in item and item[key]:
                        s_val = str(item[key])
                        items.append(s_val)
                        items.append(s_val.replace(" ", ""))
            elif isinstance(item, str):
                items.append(item)
                items.append(item.replace(" ", ""))
        return " ".join(items).lower()

    s = str(val).strip()
    if (s.startswith('[') and s.endswith(']')) or (s.startswith('{') and s.endswith('}')):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                items = []
                for item in parsed:
                    if isinstance(item, dict):
                        for key in ['name', 'title', 'character', 'job', 'director', 'original_name']:
                            if key in item and item[key]:
                                s_val = str(item[key])
                                items.append(s_val)
                                items.append(s_val.replace(" ", ""))
                    elif isinstance(item, str):
                        items.append(item)
                        items.append(item.replace(" ", ""))
                return " ".join(items).lower()
        except Exception:
            pass

    return f"{s} {s.replace(' ', '')}".lower()

# Build comprehensive searchable text blob on startup
if not new_df.empty:
    search_blobs = []
    for _, row in new_df.iterrows():
        row_tokens = []
        for col in new_df.columns:
            if col not in ['similarity', 'index', 'poster', 'movie_id', 'id']:
                parsed_text = parse_metadata_to_text(row.get(col))
                if parsed_text:
                    row_tokens.append(parsed_text)
        search_blobs.append(" ".join(row_tokens))
    new_df['search_blob'] = search_blobs
else:
    new_df['search_blob'] = []

@app.get("/")
def home():
    return {"message": "Welcome to the Cineflix Recommendation API!"}

@app.get("/app", response_class=HTMLResponse)
def serve_app():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>index.html not found in current directory</h1>", status_code=404)

@app.get("/popular")
def get_popular_movies():
    try:
        sample_df = new_df.sample(n=min(10, len(new_df)), random_state=42)
        popular_list = []
        for _, row in sample_df.iterrows():
            movie_id = int(row.get('movie_id', row.get('id', 0)))
            popular_list.append({
                "id": movie_id,
                "title": str(row.get('title', 'Untitled')),
                "poster": fetch_poster(movie_id, row.get('poster')),
                "overview": str(row.get('overview', 'No description available.')),
                "vote_average": float(row.get('vote_average', 7.5)) if pd.notna(row.get('vote_average')) else 7.5,
                "release_date": str(row.get('release_date', '2023'))[:4]
            })
        return {"movies": popular_list}
    except Exception:
        return {"movies": []}

@app.get("/catalog")
def get_catalog():
    try:
        sample_df = new_df.sample(n=min(20, len(new_df)), random_state=123)
        catalog_list = []
        for _, row in sample_df.iterrows():
            movie_id = int(row.get('movie_id', row.get('id', 0)))
            catalog_list.append({
                "id": movie_id,
                "title": str(row.get('title', 'Untitled')),
                "poster": fetch_poster(movie_id, row.get('poster')),
                "overview": str(row.get('overview', 'No description available.')),
                "vote_average": float(row.get('vote_average', 7.5)) if pd.notna(row.get('vote_average')) else 7.5,
                "release_date": str(row.get('release_date', '2023'))[:4]
            })
        return {"movies": catalog_list}
    except Exception:
        return {"movies": []}

@app.get("/search")
async def search_movies(query: str = "", filter: str = "all"):
    query_clean = re.sub(r'["\'“”]', '', query).strip().lower()
    
    if not query_clean or new_df.empty:
        return {"movies": []}

    try:
        title_col = 'title' if 'title' in new_df.columns else new_df.columns[1]
        title_series = new_df[title_col].astype(str).str.lower()
        
        q1 = query_clean
        q2 = query_clean.replace(" ", "")

        if q1 in ["sci-fi", "scifi"]:
            q1 = "science fiction"
            q2 = "sciencefiction"

        if 'search_blob' in new_df.columns:
            mask = (
                title_series.str.contains(q1, na=False) | 
                new_df['search_blob'].str.contains(q1, na=False) |
                new_df['search_blob'].str.contains(q2, na=False)
            )
        else:
            mask = title_series.str.contains(q1, na=False)

        filtered_df = new_df[mask].head(15)

        results = []
        for _, row in filtered_df.iterrows():
            movie_id = int(row.get('movie_id', row.get('id', 0)))
            poster_url = fetch_poster(movie_id, row.get('poster'))

            overview_text = str(row.get('overview', '')).strip()
            if not overview_text or overview_text.lower() in ['nan', 'none', 'null', '']:
                overview_text = str(row.get('tagline', '')).strip()
            if not overview_text or overview_text.lower() in ['nan', 'none', 'null', '']:
                movie_title = str(row.get(title_col, 'Untitled'))
                overview_text = f"'{movie_title}' is an acclaimed feature film available for discovery on Cineflix."

            results.append({
                "id": movie_id,
                "title": str(row.get(title_col, 'Untitled')),
                "poster": poster_url,
                "overview": overview_text,
                "vote_average": float(row.get('vote_average', 7.5)) if pd.notna(row.get('vote_average')) else 7.5,
                "release_date": str(row.get('release_date', '2023'))[:4] if pd.notna(row.get('release_date')) else "2023"
            })
        return {"movies": results}
    except Exception as e:
        print(f"Search error: {e}")
        return {"movies": []}
    # ==========================================
# USER AUTHENTICATION ENDPOINTS
# ==========================================

# In-memory user database (stores username -> password & security answer)
users_db = {
    "admin": {"password": "password123", "security_answer": "avatar"},
    "saniya": {"password": "San123", "security_answer": "avatar"}
}

class LoginRequest(BaseModel):
    username: str
    password: str

class SignupRequest(BaseModel):
    username: str
    password: str
    security_question: Optional[str] = "What is your favorite movie?"
    security_answer: Optional[str] = "avatar"

class ForgotPasswordRequest(BaseModel):
    username: str
    security_answer: str
    new_password: str

@app.post("/signup")
def signup(data: SignupRequest):
    user_key = data.username.strip().lower()
    if not user_key:
        return {"success": False, "message": "Username cannot be empty."}
    
    if user_key in users_db:
        return {"success": False, "message": "Username already exists. Please choose another or log in."}
    
    if len(data.password) < 4:
        return {"success": False, "message": "Password must be at least 4 characters long."}

    # Register user
    users_db[user_key] = {
        "password": data.password,
        "security_answer": data.security_answer.strip().lower() if data.security_answer else "avatar"
    }
    return {"success": True, "message": "Account created successfully! You can now log in.", "username": data.username.strip()}

@app.post("/login")
def login(data: LoginRequest):
    user_key = data.username.strip().lower()
    
    if user_key not in users_db:
        return {"success": False, "message": "User not found. Please sign up first."}
    
    if users_db[user_key]["password"] != data.password:
        return {"success": False, "message": "Incorrect password. Please try again."}

    return {"success": True, "message": "Login successful!", "username": data.username.strip()}

@app.post("/forgot-password")
def forgot_password(data: ForgotPasswordRequest):
    user_key = data.username.strip().lower()
    
    if user_key not in users_db:
        return {"success": False, "message": "Username not found."}
    
    saved_answer = users_db[user_key].get("security_answer", "").lower()
    if saved_answer and data.security_answer.strip().lower() != saved_answer:
        return {"success": False, "message": "Incorrect security answer."}

    users_db[user_key]["password"] = data.new_password
    return {"success": True, "message": "Password updated successfully!"}