# 🍼 Childcare Advanced RAG System

> An intelligent AI assistant for childcare professionals, parents, and educators powered by advanced Retrieval-Augmented Generation (RAG) technology.

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template)

## 🌟 Features

- **Advanced RAG Pipeline**: Multi-stage retrieval with query expansion and HyDE (Hypothetical Document Embeddings)
- **Real-time Processing Visualization**: Track the AI's thinking process step-by-step
- **Multi-source Knowledge Base**: Comprehensive childcare documentation and research papers
- **Web Search Integration**: Real-time information retrieval via Tavily API
- **Secure Authentication**: Built-in password protection with multi-user support
- **Conversational Interface**: Powered by Chainlit for an intuitive chat experience
- **Cloud-Ready**: Optimized Docker configuration for Railway/Render deployment

## 📋 Table of Contents

- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Deployment](#deployment)
- [API Keys Required](#api-keys-required)
- [Contributing](#contributing)

## 🏗️ Architecture

```
┌─────────────────┐
│  User Interface │  ← Chainlit Chat UI
└────────┬────────┘
         │
┌────────▼────────┐
│  RAG Integration│  ← Query Processing & Routing
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼────────┐
│Vector│  │Web Search │
│ DB   │  │  (Tavily) │
└───┬──┘  └──┬────────┘
    │        │
    └────┬───┘
         │
┌────────▼────────┐
│  LLM Generation │  ← GPT-4 / GPT-4o-mini
│  (OpenAI/Cohere)│
└─────────────────┘
```

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.12+**: Primary programming language
- **Chainlit**: Web interface and chat framework
- **LangChain**: RAG orchestration and LLM integration
- **OpenAI GPT-4**: Language model for generation
- **Cohere**: Alternative LLM provider

### Vector Database
- **Zilliz Cloud** (Production): Managed vector database
- **Milvus** (Local): Open-source vector database alternative

### Document Processing
- **Docling**: Advanced document parsing and chunking
- **Sentence Transformers**: Local embeddings (all-MiniLM-L6-v2)
- **OpenAI Embeddings**: Cloud embeddings (text-embedding-3-small)

### Additional Services
- **Tavily API**: Web search integration
- **PyTorch**: ML framework (CPU-only for deployment)

## 📁 Project Structure

```
childcare-advanced-rag-main/
├── chainlit-app/               # Main application
│   ├── app.py                  # Chainlit entry point
│   ├── rag_chainlit_integration.py  # RAG logic
│   └── chainlit.md             # Chat interface config
├── src/
│   ├── config.py               # Configuration settings
│   ├── advanced_rag/
│   │   └── processor.py        # RAG pipeline processor
│   ├── retrieval/
│   │   └── query_relevance_checker.py  # Query validation
│   └── utils/
│       └── warnings_suppressor.py  # Logging utilities
├── new_data/
│   └── qa_chunks/              # Processed document chunks
├── scripts/
│   └── test_retrieval_speed.py  # Performance testing
├── Dockerfile                  # Container configuration
├── railway.toml                # Railway deployment config
├── requirements.txt            # Python dependencies (GPU)
├── requirements-cpu.txt        # CPU-only dependencies
├── requirements-railway-pinned.txt  # Pinned versions
├── .env.example                # Environment template
└── start_app.bat               # Windows startup script
```

## 🚀 Installation

### Prerequisites
- Python 3.12 or higher
- Git
- Virtual environment (recommended)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/utsavmodi7777-oss/CHILD-CARE-RAG.git
   cd CHILD-CARE-RAG
   ```

2. **Create virtual environment**
   ```bash
   # Windows
   python -m venv .venv_win
   .venv_win\Scripts\activate

   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   # For local development (CPU-only)
   pip install -r requirements-cpu.txt

   # For GPU support
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

5. **Run the application**
   ```bash
   # Windows
   start_app.bat

   # Linux/Mac
   cd chainlit-app
   chainlit run app.py --watch
   ```

6. **Access the application**
   Open your browser at: `http://localhost:8000`

## ⚙️ Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Required API Keys
OPENAI_API_KEY=sk-...                    # OpenAI API key
TAVILY_API_KEY=tvly-...                  # Tavily search API key
COHERE_API_KEY=...                       # Cohere API key (optional)

# Vector Database (Choose one)
ZILLIZ_URI=https://...                   # Zilliz Cloud URI
ZILLIZ_TOKEN=...                         # Zilliz API token

# Authentication
CHAINLIT_AUTH_SECRET=your_secret_key_here
AUTH_USER_1=admin:strongpassword123
AUTH_USER_2=user:password456

# Model Configuration
OPENAI_MODEL=gpt-4o-mini                 # or gpt-4
EMBEDDING_MODEL=text-embedding-3-small

# Optional Settings
USE_GPU=false                            # Set to true for GPU support
TOP_K_RETRIEVAL=5
WEB_SEARCH_ENABLED=true
```

## 🔑 API Keys Required

You'll need to sign up for these services:

1. **OpenAI** (Required): https://platform.openai.com/
   - Models: GPT-4, GPT-4o-mini, text-embedding-3-small

2. **Tavily** (Required): https://tavily.com/
   - Web search functionality

3. **Zilliz Cloud** (Required): https://cloud.zilliz.com/
   - Managed vector database
   - Free tier available

4. **Cohere** (Optional): https://cohere.com/
   - Alternative LLM provider

## 📖 Usage

### Basic Chat
1. Log in with credentials from your `.env` file
2. Ask questions about childcare, parenting, or child development
3. Watch the AI process your query in real-time

### Example Queries
- "What are the developmental milestones for a 2-year-old?"
- "How can I manage tantrums in toddlers?"
- "What are effective potty training strategies?"
- "Explain social-emotional development in preschoolers"

### Advanced Features
- **Web Search**: Automatically triggers for current events or trending topics
- **Source Citations**: See which documents informed the AI's response
- **Processing Steps**: View query expansion, retrieval, and generation stages

## 🌐 Deployment

### Deploy to Railway

1. **Create Railway account**: https://railway.app/

2. **Create new project from GitHub**
   - Connect your repository
   - Railway auto-detects the Dockerfile

3. **Add environment variables**
   - Go to Variables tab
   - Add all variables from your `.env` file

4. **Deploy**
   - Railway automatically builds and deploys
   - Get your public URL: `https://your-app.railway.app`

### Deploy to Render

1. **Create Render account**: https://render.com/

2. **New Web Service**
   - Connect GitHub repository
   - Select "Docker" as environment

3. **Configure**
   - Add environment variables
   - Set health check path: `/`

4. **Deploy**
   - Render builds from Dockerfile
   - Get your URL: `https://your-app.onrender.com`

### Deploy to Hugging Face Spaces

1. **Create Space**: https://huggingface.co/spaces
2. **Select Docker SDK**
3. **Upload files and Dockerfile**
4. **Add secrets in Settings**

## 🧪 Testing

Run the comprehensive system test:
```bash
python complete_system_test.py
```

Test retrieval performance:
```bash
python scripts/test_retrieval_speed.py
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Utsav Modi**
- GitHub: [@utsavmodi7777-oss](https://github.com/utsavmodi7777-oss)
- Repository: [CHILD-CARE-RAG](https://github.com/utsavmodi7777-oss/CHILD-CARE-RAG)

## 🙏 Acknowledgments

- Childcare professionals and educators who contributed domain knowledge
- Open-source community for amazing tools and libraries
- Research papers and textbooks that formed the knowledge base

## 📧 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: [Your Email]

---

**Made with ❤️ for childcare professionals and parents**
