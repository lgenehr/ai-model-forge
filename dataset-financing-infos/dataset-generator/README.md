# Dataset Generator

Multi-source dataset generator for LLM fine-tuning, focused on Portuguese (Brazilian) content.

## Features

- **Multiple Data Sources**: News, Wikipedia, Books, Academic papers, Legal documents, Social media, Videos
- **10 Topic Categories**: Finance, Technology, Science, Health, Legal, Humanities, Culture, Business, Education, Environment
- **Advanced Processing**: Text cleaning, deduplication (MinHash), quality filtering
- **Safety Guardrails**: PII detection, content filtering, bias checking, language detection
- **Multiple Output Formats**: Alpaca, ShareGPT, ChatML
- **Resumable Collection**: Checkpoint system for long-running collections
- **Rich CLI**: Beautiful terminal interface with progress tracking

## Installation

```bash
# Clone the repository
git clone https://github.com/ai-model-forge/dataset-generator
cd dataset-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .

# Or install with all optional dependencies
pip install -e ".[full]"
```

## Configuration

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Required API Keys

- **NewsAPI** (free tier: 100 requests/day): https://newsapi.org/
- **YouTube Data API** (free tier: 10,000 units/day): https://console.cloud.google.com/
- **Reddit API** (60 requests/minute): https://www.reddit.com/prefs/apps
  - Use `REDDIT_ACCESS_TOKEN` when running inside Devvit to authenticate with the token provided by Reddit.

### Optional API Keys

- **Semantic Scholar** (academic papers): https://www.semanticscholar.org/product/api
- **GNews** (additional news): https://gnews.io/

## Usage

### Quick Start

```bash
# Collect data from all sources for finance and technology topics
python -m src.main collect --sources all --topics financeiro,tecnologia --max-items 100

# Process collected data
python -m src.main process --quality-threshold 0.7

# Format for Alpaca
python -m src.main format --formats alpaca

# View statistics
python -m src.main stats
```

### CLI Commands

#### `collect` - Collect data from sources

```bash
python -m src.main collect \
    --sources news,encyclopedia,books \
    --topics financeiro,tecnologia \
    --max-items 500 \
    --resume \
    --workers 4
```

Options:
- `--sources, -s`: Data sources (news, encyclopedia, books, academic, legal, social_media, videos, or "all")
- `--topics, -t`: Topics (financeiro, tecnologia, ciencias, saude, juridico, humanidades, cultura, negocios, educacao, meio_ambiente, or "all")
- `--max-items, -m`: Maximum items per source/topic combination
- `--resume/--no-resume`: Resume from checkpoint
- `--workers, -w`: Number of parallel workers
- `--output-dir, -o`: Output directory

#### `process` - Clean and filter data

```bash
python -m src.main process \
    --input-dir ./outputs/raw \
    --output-dir ./outputs/processed \
    --quality-threshold 0.7 \
    --deduplicate \
    --remove-pii
```

Options:
- `--quality-threshold, -q`: Minimum quality score (0-1)
- `--deduplicate/--no-deduplicate`: Remove duplicate entries
- `--remove-pii/--no-remove-pii`: Remove personally identifiable information
- `--filter-content/--no-filter-content`: Filter inappropriate content

#### `format` - Format for training

```bash
python -m src.main format \
    --input-dir ./outputs/processed \
    --output-dir ./outputs/final \
    --formats alpaca,sharegpt,chatml \
    --split-ratio 0.9,0.05,0.05
```

Options:
- `--formats, -f`: Output formats (alpaca, sharegpt, chatml)
- `--split-ratio`: Train/validation/test split ratio

#### `stats` - View statistics

```bash
python -m src.main stats --input-dir ./outputs/final
```

#### `validate` - Validate dataset quality

```bash
python -m src.main validate --input-dir ./outputs/final --sample-size 200
```

## Data Sources

| Source | Description | API Key Required |
|--------|-------------|------------------|
| `news` | RSS feeds from Brazilian news portals | Optional (NewsAPI) |
| `encyclopedia` | Wikipedia PT-BR articles | No |
| `books` | Project Gutenberg, Open Library | No |
| `academic` | arXiv, Semantic Scholar | Optional |
| `legal` | Brazilian legislation (Planalto, STF) | No |
| `social_media` | Reddit (Brazilian subreddits) | Yes |
| `videos` | YouTube metadata and descriptions | Yes |

## Topics

| Topic | Description | Subtopics |
|-------|-------------|-----------|
| `financeiro` | Finance and economy | Economia, Mercado, Investimentos, Cripto, Banking |
| `tecnologia` | Technology and computing | Programação, IA, Data Science, Security, Cloud |
| `ciencias` | Natural sciences | Física, Química, Biologia, Matemática, Astronomia |
| `saude` | Health and medicine | Medicina, Nutrição, Psicologia, Fitness |
| `juridico` | Law and legislation | Civil, Penal, Trabalhista, Tributário, Constitucional |
| `humanidades` | Humanities | História, Filosofia, Sociologia, Antropologia, Política |
| `cultura` | Culture and arts | Literatura, Arte, Música, Cinema, Gastronomia |
| `negocios` | Business | Empreendedorismo, Marketing, Gestão, RH, Vendas |
| `educacao` | Education | Pedagogia, Ensino Superior, Educação Infantil |
| `meio_ambiente` | Environment | Sustentabilidade, Clima, Energia, Conservação |

## Output Formats

### Alpaca Format

```json
{
    "instruction": "Explique sobre economia brasileira.",
    "input": "",
    "output": "A economia brasileira é caracterizada por...",
    "metadata": {
        "source": "news",
        "topic": "financeiro",
        "quality_score": 0.85
    }
}
```

### ShareGPT Format

```json
{
    "conversations": [
        {"from": "human", "value": "Explique sobre economia brasileira."},
        {"from": "gpt", "value": "A economia brasileira é caracterizada por..."}
    ]
}
```

### ChatML Format

```json
{
    "messages": [
        {"role": "system", "content": "Você é um especialista em finanças..."},
        {"role": "user", "content": "Explique sobre economia brasileira."},
        {"role": "assistant", "content": "A economia brasileira é caracterizada por..."}
    ]
}
```

## Quality Filtering

The generator applies multiple quality filters:

1. **Length Filter**: 50-4096 tokens
2. **Quality Score**: Text quality metrics (0.6+ to pass)
3. **Deduplication**: MinHash with 0.85 similarity threshold
4. **Language Detection**: 95% confidence for Portuguese
5. **Content Filter**: Removes offensive/spam/dangerous content
6. **PII Detection**: Removes CPF, phone, email, addresses

## Project Structure

```
dataset-generator/
├── src/
│   ├── main.py              # CLI entry point
│   ├── config/              # Configuration files
│   ├── collectors/          # Data collection modules
│   ├── processors/          # Data processing modules
│   ├── guardrails/          # Safety and quality checks
│   ├── storage/             # Checkpoint and output management
│   ├── utils/               # Utilities (logging, retry, etc.)
│   └── schemas/             # Pydantic data models
├── outputs/                 # Generated datasets
├── checkpoints/             # Collection checkpoints
├── logs/                    # Log files
└── tests/                   # Unit tests
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
ruff check src/
```

## Target Dataset Size

For fine-tuning 7B-14B parameter models:

| Target | Samples | Tokens |
|--------|---------|--------|
| Minimum | 50,000 | 50M |
| Recommended | 200,000 | 200M |
| Optimal | 500,000 | 500M |

## License

MIT License
