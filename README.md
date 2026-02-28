# Hybrid Recommender System

Streamlit-based movie recommendation application. Combines user-based, item-based, and content-based approaches into a hybrid system.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run
streamlit run app.py

# Docker
docker-compose up
```

Application: `http://localhost:8080`

## Project Structure

```
├── app.py                 # Streamlit UI
├── config.py              # Configuration
├── utils.py               # Utilities
├── error_handling.py      # Error handling
├── logging_config.py      # Logging
├── security_utils.py      # Security
├── performance_utils.py   # Performance
├── data_loader/           # Data loading
├── recommenders/          # Algorithms
├── ui/                    # UI components
└── tests/                 # Tests (115 tests, 70% coverage)
```

## Application Flow Diagram

```mermaid
flowchart TD
    A[User opens Streamlit app] --> B[Load config and pickle data]
    B --> C[Render UI and input controls]
    C --> D[User selects recommendation mode and parameters]
    D --> E{Recommendation mode}

    E -->|User-Based| F[Find similar users via Pearson correlation]
    F --> F1[Filter by overlap ratio, corr threshold, max neighbors]
    F1 --> F2[Compute weighted scores and rank unseen movies]

    E -->|Item-Based| G[Get user's latest 5-star movie]
    G --> G1[Compute movie-movie correlation]
    G1 --> G2[Return top-N similar movies]

    E -->|Content-Based| H[Build/Use genre vectors]
    H --> H1[Compute cosine similarity on genres]
    H1 --> H2[Return top-N similar movies]

    E -->|Hybrid| I[Run User-Based + Item-Based + Content-Based]
    I --> I1[Combine candidates by movie title]
    I1 --> I2[Compute Source_Count and Average_Score]
    I2 --> I3[Hybrid_Confidence = Source_Count * Average_Score]
    I3 --> I4[Sort and return top-N hybrid recommendations]

    F2 --> Z[Display recommendations and debug metrics]
    G2 --> Z
    H2 --> Z
    I4 --> Z
```

## Development

```bash
# Format & lint
make format
make lint

# Tests
make test              # Unit tests
make test-integration  # Integration tests
make test-all          # All tests
make test-cov          # With coverage

# Pre-commit
make setup-precommit
```

## Configuration

Environment variables:
- `PICKLE_PATH` - Data file path
- `LOG_LEVEL` - DEBUG|INFO|WARNING|ERROR
- `SERVER_PORT` - Default: 8080

## Testing

115 tests, 70% coverage:
- 84 unit tests
- 16 integration tests
- 15 utility/UI tests

## Deployment

```bash
# Docker
docker-compose up --build

# View logs
docker-compose logs -f
```

Prebuilt image deployment (GHCR):

```bash
docker compose -f docker-compose.ghcr.yml pull
docker compose -f docker-compose.ghcr.yml up -d
```

## Tech Stack

- Python 3.11
- Streamlit 1.50.0
- Pandas, NumPy, Scikit-learn
- pytest (testing)
- Docker (deployment)
