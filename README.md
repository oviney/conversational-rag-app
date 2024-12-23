# Conversational RAG Application

This repository contains a fully implemented Retrieval-Augmented Generation (RAG) application. The project is built with Python, Streamlit, and Docker, and is designed to demonstrate how GenAI can be incorporated into the Software Development Lifecycle (SDLC) for tasks such as generating insights from business requirements, test case generation, and feature refinement.

## **Features**

### **Core Functionality**
- **Document Upload**: Supports PDF, DOCX, and TXT file formats.
- **Text Extraction and Preprocessing**: Extracts and preprocesses text for further analysis.
- **Chunking and Vector Embedding**: Uses Sentence Transformers to chunk and embed document text.
- **In-memory FAISS Index**: For fast vector storage and retrieval.
- **Text Generation**: Leverages GPT-2 via Hugging Face Transformers to generate responses.

### **UI and Interaction**
- A user-friendly **Streamlit** interface for document upload, querying, and chat-based interactions.

### **DevSecOps Practices**
- GitHub Actions for **CI/CD pipelines**.
- Static code analysis using **Bandit**.
- Containerization with **Docker** for portability.
- Testing framework with **pytest**.

## **Project Structure**

```
rag_project/
├── app/
│   ├── config.py             # Centralized configurations
│   ├── services/             # Service modules for core logic
│   │   ├── document_service.py
│   │   ├── retrieval_service.py
│   │   ├── generation_service.py
│   ├── document_processing.py
│   ├── utils.py              # Utility functions
├── tests/                    # Unit and integration tests
├── streamlit_app.py          # Main Streamlit UI application
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker setup
├── README.md                 # Documentation
└── .github/
    └── workflows/
        └── devsecops.yml     # CI/CD workflow
```

## **Setup and Installation**

### **Prerequisites**
- Python 3.9 or later
- Docker (optional, for containerization)

### **Steps to Run Locally**
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd rag_project
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Application**:
   ```bash
   streamlit run streamlit_app.py
   ```

### **Using Docker**
1. **Build the Docker Image**:
   ```bash
   docker build -t rag-app .
   ```
2. **Run the Docker Container**:
   ```bash
   docker run -p 8501:8501 rag-app
   ```

### **Run Tests**
```bash
pytest tests/
```

## **How It Works**

1. **Upload a Document**: The user uploads a document via the Streamlit UI.
2. **Text Processing**: The document is processed to extract text, preprocess it, and chunk it into manageable pieces.
3. **Vector Storage**: Text chunks are converted into embeddings and stored in an in-memory FAISS index.
4. **Querying**: Users enter a query, and the application retrieves relevant chunks from the FAISS index.
5. **Text Generation**: The retrieved context is passed to a Hugging Face model (default: GPT-2) to generate a response.

## **DevSecOps Workflow**

This project includes a CI/CD pipeline using GitHub Actions:
- **Linting and Code Quality**: Uses `Bandit` for static code analysis.
- **Unit Tests**: Ensures correctness of individual components.
- **Docker Build and Test**: Builds and tests the container for deployment readiness.

### Sample Workflow File
```yaml
name: DevSecOps Workflow

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  build-and-test:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: pip install -r requirements.txt

    - name: Run tests
      run: pytest tests/

    - name: Lint with Bandit
      run: bandit -r app/

    - name: Build Docker image
      run: docker build -t rag-app .
```

## **Use Cases in the SDLC**
- **Business Analysts**: Process BRDs and extract features or user stories.
- **Developers**: Generate boilerplate code from design documents.
- **Testers**: Create test cases or automate test scripts based on requirements.
- **Architects**: Validate design decisions using conversational queries.

## **Contributing**
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.
