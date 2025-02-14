# Movie Recommendation System

A sophisticated movie recommendation engine built with Python and FastAPI, leveraging machine learning techniques to provide personalized movie suggestions based on user input.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.112.0-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.1-orange)

## Features

- **Advanced Recommendation Engine**: Utilizes cosine similarity on precomputed movie embeddings for accurate movie suggestions
- **Modern UI/UX**: Responsive design with dark/light mode support
- **Real-time Recommendations**: Instant movie suggestions with similarity scores
- **Interactive Interface**: Modal views for detailed movie information
- **Mobile-Friendly**: Fully responsive design that works on all devices

## Technical Implementation

### Backend
- **FastAPI Framework**: High-performance web framework for building APIs
- **Scikit-learn**: Used for computing cosine similarity between movie embeddings
- **Numpy & Pandas**: Efficient data manipulation and numerical computations
- **Type Hints**: Enhanced code readability and maintainability
- **Caching**: LRU cache implementation for improved performance
- **Error Handling**: Comprehensive error handling with proper HTTP status codes

### Frontend
- **Modern Design**: Clean and intuitive user interface
- **Dark/Light Mode**: Theme toggle for better user experience
- **Responsive Layout**: CSS Grid and Flexbox for responsive design
- **Loading States**: Visual feedback during API calls
- **Error Handling**: User-friendly error messages
- **Interactive Elements**: Smooth animations and transitions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/movie-recommendation.git
cd movie-recommendation
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
uvicorn app:app --reload
```

5. Open your browser and navigate to `http://localhost:8000`

## Project Structure

```
movie-recommendation/
├── app.py                 # FastAPI application and API endpoints
├── static/
│   └── styles.css        # CSS styles and theme configuration
├── templates/
│   └── index.html        # Frontend template
├── movie_embeddings.npy  # Precomputed movie embeddings
├── movie_metadata.csv    # Movie metadata
├── requirements.txt      # Project dependencies
├── .gitignore           # Git ignore configuration
├── LICENSE              # MIT License
└── README.md            # Project documentation
```

## Technical Details

### Recommendation Algorithm

The system uses the following approach to generate recommendations:

1. **Embedding Generation**: Movie features are converted into numerical embeddings
2. **Similarity Computation**: Cosine similarity is used to measure movie similarities
3. **Ranking**: Movies are ranked based on similarity scores
4. **Filtering**: Input movie is excluded from recommendations

### Performance Optimization

- LRU caching for frequently requested recommendations
- Precomputed embeddings for faster similarity calculations
- Efficient data structures for quick lookups
- Optimized database queries and data loading

## Future Improvements

- [ ] Integration with TMDB API for movie posters
- [ ] User authentication and personalized recommendations
- [ ] Advanced filtering options (by genre, year, rating)
- [ ] Recommendation explanations
- [ ] Performance metrics and analytics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 