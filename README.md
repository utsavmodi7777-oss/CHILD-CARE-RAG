# Childcare Advanced RAG System

AI-powered Question-Answering assistant for child development, parenting, and early childhood education using advanced Retrieval-Augmented Generation (RAG) techniques.

## 🌟 Overview

This system provides intelligent responses to questions about child care, development, behavior management, and parenting using a curated knowledge base of 3,593 educational document chunks from 15 authoritative sources.

**Live Demo:** Access at `http://localhost:8000` after setup

## ⚙️ Technology Stack

- **Framework:** LangChain + Chainlit (Web UI)
- **Embeddings:** Google Gemini API (text-embedding-004, 768 dimensions)
- **Vector Database:** Zilliz Cloud (Serverless)
- **Python:** 3.11.9
- **RAG Technique:** CRAG (Corrective Retrieval-Augmented Generation)
- **Reranking:** Cohere Rerank API

## 📁 Project Structure

```
childcare-advanced-rag-main/
├── chainlit-app/
│   ├── app.py                          # Main Chainlit web application
│   └── rag_chainlit_integration.py     # RAG system integration
├── src/
│   ├── advanced_rag/
│   │   ├── pipeline.py                 # RAG pipeline orchestration
│   │   └── processor.py                # Document processing
│   ├── config/
│   │   ├── logging_config.py           # Logging configuration
│   │   ├── settings.py                 # Environment settings (Pydantic)
│   │   └── vector_config.py            # Vector DB configuration
│   ├── data_processing/
│   │   ├── docling_processor.py        # PDF processing with Docling
│   │   ├── embedding_generator.py      # Embedding generation
│   │   └── process_all_pdfs_optimized.py
│   ├── evaluation/
│   │   ├── answer_enhancer.py          # Answer quality enhancement
│   │   ├── answer_evaluator.py         # Evaluation metrics
│   │   └── retrieval_assessor.py       # Retrieval quality assessment
│   ├── generation/
│   │   └── enhanced_crag.py            # CRAG implementation
│   ├── retrieval/
│   │   ├── cohere_reranking.py         # Cohere reranker
│   │   ├── hyde_generation.py          # HyDE (Hypothetical Document Embeddings)
│   │   └── query_expansion.py          # Query expansion techniques
│   ├── utils/
│   │   └── client_manager.py           # API client management
│   └── vector_store/
│       └── zilliz_manager.py           # Zilliz operations
├── new_data/
│   └── qa_chunks/                      # 15 JSON files with 3,593 chunks
├── generate_gemini_embeddings.py       # Embeddings generation script
├── requirements.txt                     # Python dependencies
├── .env                                # Environment variables (not in repo)
└── README.md                           # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11.x
- Google Gemini API Key (Free tier: https://aistudio.google.com/app/apikey)
- Zilliz Cloud account (Free tier: https://cloud.zilliz.com/)
- Cohere API Key (Free tier: https://cohere.com/)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/utsavmodi7777-oss/CHILD-CARE-RAG.git
cd CHILD-CARE-RAG
```

2. **Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
# OR
source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create `.env` file in root directory:
```env
# Google Gemini API (Free - No Credit Card Required)
GEMINI_API_KEY=your_gemini_api_key_here

# Zilliz Cloud (Vector Database)
ZILLIZ_URI=your_zilliz_uri_here
ZILLIZ_TOKEN=your_zilliz_token_here

# Cohere API (Reranking)
COHERE_API_KEY=your_cohere_api_key_here

# Chainlit Authentication (Optional)
AUTH_USER_1=admin:admin123
AUTH_USER_2=user:password
```

5. **Generate embeddings** (First time only, ~45 minutes)
```bash
python generate_gemini_embeddings.py
```

This will:
- Load 3,593 text chunks from `new_data/qa_chunks/`
- Generate 768-dimensional embeddings using Gemini
- Create Zilliz collection `childcare_knowledge_base`
- Insert embeddings into vector database

6. **Start the application**
```bash
cd chainlit-app
chainlit run app.py
```

7. **Access the UI**
Open browser: http://localhost:8000

## 🔑 Getting Free API Keys

### Google Gemini API (Embeddings)
1. Visit: https://aistudio.google.com/app/apikey
2. Sign in with Google account
3. Click "Create API Key"
4. Copy the key to `.env` as `GEMINI_API_KEY`

**Limits:** 15 requests/minute (Free tier)

### Zilliz Cloud (Vector Database)
1. Visit: https://cloud.zilliz.com/
2. Sign up for free account
3. Create new serverless cluster
4. Copy URI and Token to `.env`

**Limits:** 100MB storage (Free tier)

### Cohere API (Reranking)
1. Visit: https://cohere.com/
2. Sign up for free account
3. Get API key from dashboard
4. Copy to `.env` as `COHERE_API_KEY`

**Limits:** 100 API calls/month (Free tier)

## 📊 Knowledge Base

The system contains **3,593 curated text chunks** from 15 authoritative sources:

1. Child Development OER Textbook
2. Principles of Child Development
3. Early Childhood Education guides
4. Social-Emotional Development resources
5. Parenting handbooks
6. Behavior management guides
7. Infant-toddler development materials
8. Preschool education resources
9. Parent-child interaction studies

**Topics Covered:**
- Physical, cognitive, and emotional development
- Age-appropriate parenting strategies
- Behavior management techniques
- Early learning and education
- Social-emotional development
- Nutrition and health
- Safety and child protection

## 🧠 RAG Architecture

### Retrieval Pipeline
1. **Query Processing**
   - Query expansion (generate similar queries)
   - HyDE generation (create hypothetical documents)

2. **Vector Search**
   - Semantic search in Zilliz (top-k retrieval)
   - 768-dimensional Gemini embeddings

3. **Reranking**
   - Cohere Rerank API for relevance scoring
   - Select top-3 most relevant chunks

4. **CRAG (Corrective RAG)**
   - Assess retrieval quality
   - Decide: Use retrieved docs / Search web / Generate directly
   - Enhance answer with context

### Generation
- LangChain integration
- Context-aware response generation
- Source attribution

## 🛠️ Development

### Add New Documents
1. Place PDF in `data/` folder
2. Run processing script:
```bash
python src/data_processing/docling_processor.py
```
3. Regenerate embeddings:
```bash
python generate_gemini_embeddings.py
```

### Evaluation
Run evaluation suite:
```bash
python src/evaluation/answer_evaluator.py
```

Metrics:
- Retrieval accuracy
- Answer relevance
- Factual consistency
- Source attribution

## 📝 Configuration

### Vector Database (`src/config/vector_config.py`)
```python
embedding_dimension = 768  # Gemini text-embedding-004
collection_name = "childcare_knowledge_base"
```

### Settings (`src/config/settings.py`)
- API keys management
- Logging configuration
- Authentication settings

## 🐛 Troubleshooting

### "Collection not found" error
```bash
python generate_gemini_embeddings.py
```

### Rate limit errors (429)
- Gemini: Wait 1 minute between batches
- Cohere: Use free tier limits

### Module not found
```bash
pip install -r requirements.txt --force-reinstall
```

## 📦 Dependencies

**Core:**
- `chainlit==1.3.3` - Web UI framework
- `langchain-core==0.3.81` - RAG framework
- `google-generativeai==0.8.6` - Gemini embeddings
- `pymilvus==2.4.9` - Zilliz client
- `langchain-cohere==0.4.6` - Cohere integration

**Full list:** See `requirements.txt`

## 🔒 Security Notes

- **Never commit `.env` file** - Contains API keys
- Add `.env` to `.gitignore`
- Use environment variables for secrets
- Implement Chainlit authentication for production

## 📄 License

This project is for educational purposes. Ensure compliance with source document licenses.

## 👨‍💻 Author

**Utsav Modi**
- GitHub: [@utsavmodi7777-oss](https://github.com/utsavmodi7777-oss)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📧 Support

For issues and questions, please open an issue on GitHub.

## 🙏 Acknowledgments

- Educational content sources (see `new_data/qa_chunks/`)
- LangChain community
- Chainlit framework
- Google Gemini API
- Zilliz Cloud
- Cohere

---

**⭐ Star this repo if you find it helpful!**
