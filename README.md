# Hybrid Recommender System

A comprehensive hybrid movie recommendation system that combines User-Based, Item-Based, and Content-Based recommendation approaches.

## Features

- **User-Based Recommendation**: Find similar users and recommend films they liked
- **Item-Based Recommendation**: Recommend films similar to user's favorite films
- **Content-Based Recommendation**: Suggest films based on genre similarity
- **Hybrid Approach**: Combine all three methods for better accuracy

## Tech Stack

- **Streamlit**: Web application framework
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: Machine learning (TF-IDF, Cosine Similarity)
- **Docker**: Containerization for easy deployment

## Installation

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/cihanoguz/hybrid-recommender-project.git
cd hybrid-recommender-project
```

2. Create virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Prepare data (generate pickle files):
```bash
cd data
python prepare_data.py
python shrink_dataset.py
```

4. Run the application:
```bash
streamlit run app.py --server.port=8080
```

### Docker Deployment

1. Build and run with Docker Compose:
```bash
docker-compose up -d --build
```

2. Access the application at `http://localhost:8080`

## Project Structure

```
hybrid-recommender-project/
├── app.py                          # Main Streamlit application
├── data/
│   ├── prepare_data.py            # Data preprocessing script
│   ├── shrink_dataset.py          # Dataset reduction for demo
│   ├── movie.csv                  # Movie dataset (ignored in git)
│   ├── rating.csv                 # Rating dataset (ignored in git)
│   ├── prepare_data.pkl           # Processed data (ignored in git)
│   └── prepare_data_demo.pkl      # Demo dataset (ignored in git)
├── HYBRID_RECOMMENDER_PROJECT-tutorial.py  # Tutorial/learning script
├── Dockerfile                     # Docker image definition
├── docker-compose.yml             # Docker Compose configuration
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Dataset

This project uses the MovieLens dataset:
- **Movies**: 27,278 films
- **Ratings**: 20,000,263+ ratings
- **Users**: 138,493 users
- **Time Range**: 1995-2015

**Note**: Large data files (CSV and pickle files) are excluded from the repository. You need to download the MovieLens dataset and process it using `data/prepare_data.py`.

## Usage

1. Select a target user ID (default: 108170)
2. Choose recommendation method:
   - User-Based: Find users with similar taste
   - Item-Based: Find films similar to user's favorites
   - Content-Based: Find films with similar genres
   - Hybrid: Combine all three approaches
3. Adjust parameters (overlap percentage, correlation threshold, etc.)
4. View recommendations and detailed analysis

## Documentation

- `DOCKER_SETUP.md`: Docker setup and deployment guide
- `LINUX_DEPLOYMENT.md`: Linux server deployment guide
- `QUICK_START_LINUX.md`: Quick start guide for Linux
- `DEBIAN_BUILD_SUMMARY.md`: Debian build process summary

## License

This project is for educational purposes.

## Author

[Cihan Oguz](https://github.com/cihanoguz)

