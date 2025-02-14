from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import ast
import logging
from typing import List, Dict, Optional
from pydantic import BaseModel
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI with metadata
app = FastAPI(
    title="Movie Recommendation System",
    description="A sophisticated movie recommendation system using cosine similarity and movie embeddings",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the precomputed embeddings and metadata
try:
    embedding_matrix = np.load('movie_embeddings.npy')
    movies_cleaned = pd.read_csv('movie_metadata.csv')
    logger.info("Successfully loaded movie embeddings and metadata")
except Exception as e:
    logger.error(f"Error loading data: {str(e)}")
    raise

class MovieRecommendation(BaseModel):
    title: str
    genres: str
    overview: str
    similarity_score: float

def extract_genre_names(genre_list: str) -> str:
    try:
        if isinstance(genre_list, str):
            genre_list = ast.literal_eval(genre_list)
        genre_names = [genre['name'] for genre in genre_list]
        return ', '.join(genre_names)
    except Exception as e:
        logger.error(f"Error extracting genre names: {str(e)}")
        return ""

# Apply the function to the DataFrame
movies_cleaned['genres'] = movies_cleaned['genres'].apply(extract_genre_names)

# Calculate the cosine similarity matrix
cosine_sim = cosine_similarity(embedding_matrix, embedding_matrix)

# Setup templates and static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@lru_cache(maxsize=100)
def recommend_movies_by_title(title: str, n: int = 21) -> List[Dict]:
    try:
        # Case-insensitive search for movies
        matching_movies = movies_cleaned[movies_cleaned['title'].str.contains(title, case=False)]
        
        if matching_movies.empty:
            raise HTTPException(
                status_code=404,
                detail="No movies found with that title. Please try another movie name."
            )
        
        # Get the first matching movie
        idx = matching_movies.index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Exclude the input movie from recommendations
        sim_scores = [score for score in sim_scores if score[0] != idx]
        
        if not sim_scores:
            raise HTTPException(
                status_code=404,
                detail="Couldn't find any similar movies. Please try another title."
            )
            
        movie_indices = [i[0] for i in sim_scores[:n]]
        
        recommendations = []
        for i, movie_idx in enumerate(movie_indices):
            movie = movies_cleaned.iloc[movie_idx]
            recommendations.append({
                'title': movie['title'],
                'genres': movie['genres'],
                'overview': movie['overview'],
                'similarity_score': round(sim_scores[i][1] * 100, 2)
            })
        
        return recommendations
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while finding movie recommendations. Please try again later."
        )

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Render the main page of the movie recommendation system.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/recommend", response_model=List[MovieRecommendation])
async def recommend(title: str = Form(...)):
    """
    Get movie recommendations based on a given title.
    
    Parameters:
    - title: The title of the movie to base recommendations on
    
    Returns:
    - List of movie recommendations with similarity scores
    """
    try:
        if not title.strip():
            raise HTTPException(
                status_code=400,
                detail="Please enter a movie title to get recommendations."
            )
        recommendations = recommend_movies_by_title(title)
        return recommendations
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in recommend endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later."
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
