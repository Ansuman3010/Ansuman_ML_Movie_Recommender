import pickle
import streamlit as st
import requests
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = st.secrets["TMDB_API_KEY"]

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"

)

st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba");
        background-size: cover;
        
    }
    .movie-card {
        text-align: center;
        
        border-radius: 10px;
        background-color: #06141B;
    }
    .movie-title {
        font-size: 16px;
        font-weight: bold;
        
    }
    </style>
""", unsafe_allow_html=True)

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
    except requests.exceptions.RequestException as e:
        print("Error fetching poster:", e)

    return None
movies = pickle.load(open('movie.pkl', 'rb'))
def compute_similarity(data):
    tfidf = TfidfVectorizer(max_features=5000,stop_words="english")

    vectors = tfidf.fit_transform(data["tags"]).toarray()
    similarity_matrix = cosine_similarity(vectors)

    return similarity_matrix

similarity = compute_similarity(movies)
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]

    distances = sorted(list(enumerate(similarity[index])),reverse=True,key=lambda x: x[1])

    top_similar = distances[1:15]

    top_similar = sorted(top_similar,key=lambda x: movies.iloc[x[0]].weighted_rating,reverse=True)

    movie_names = []
    movie_posters = []
    movie_ratings = []

    for i in top_similar[:5]:
        movie_id = movies.iloc[i[0]].id
        movie_posters.append(fetch_poster(movie_id))
        movie_names.append(movies.iloc[i[0]].title)
        movie_ratings.append(round(movies.iloc[i[0]].weighted_rating, 2))
        time.sleep(0.2)

    return movie_names, movie_posters, movie_ratings




st.markdown(
    "<h1 style='text-align:center; color:#CCD0CF;'>🎬 Movie Recommender System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Find movies similar to your favorite ones</p>",
    unsafe_allow_html=True
)

st.divider()

col1, col2, col3 = st.columns([1,2,1])

with col2:
    selected_movie = st.selectbox(
        "Select a Movie",
        movies['title'].values
    )

    recommend_btn = st.button("🔍 Show Recommendations", use_container_width=True)

if recommend_btn:
    with st.spinner("Finding best movies for you... 🍿"):
        movie_names, movie_posters, movie_ratings = recommend(selected_movie)

    st.divider()
    st.subheader("Recommended Movies")

    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            st.markdown("<div class='movie-card'>", unsafe_allow_html=True)

            if movie_posters[i]:
                st.image(movie_posters[i])

            st.markdown(
                f"<div class='movie-title'>{movie_names[i]}</div>",
                unsafe_allow_html=True
            )

            st.caption(f"Rating: {movie_ratings[i]}")

            st.markdown("</div>", unsafe_allow_html=True)
