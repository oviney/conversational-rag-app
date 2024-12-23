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

## **Architectural Choices and Design Decisions**

### **1. Modular Architecture**
- **Purpose**: Ensure each functionality is isolated, making the codebase maintainable and extendable.
- **Implementation**: 
  - Separate modules for configuration (`config.py`), services (`document_service.py`, `retrieval_service.py`, `generation_service.py`), and utilities (`utils.py`).
  - Streamlit handles the UI independently, and each service can function autonomously.
- **Benefit**: Facilitates testing and debugging by decoupling responsibilities.

### **2. Choice of FAISS for Vector Storage**
- **Purpose**: Provide fast, in-memory vector similarity search.
- **Implementation**: FAISS is used to index document embeddings and retrieve the most relevant chunks.
- **Benefit**: It is highly efficient for approximate nearest neighbor search, enabling real-time interaction for RAG workflows.

### **3. Sentence Transformers for Embedding**
- **Purpose**: Generate high-quality vector embeddings from text chunks.
- **Implementation**: The `sentence-transformers` library is used for chunk embedding generation.
- **Benefit**: Optimized for semantic similarity, improving the accuracy of retrieval and the relevance of responses.

### **4. Hugging Face Transformers for Text Generation**
- **Purpose**: Provide conversational capabilities using a pre-trained language model.
- **Implementation**: GPT-2 is used for generating responses based on the retrieved chunks and user queries.
- **Benefit**: Integrates well with Hugging Face’s pipeline API, ensuring extensibility with other models.

### **5. Streamlit for User Interaction**
- **Purpose**: Create an intuitive and lightweight interface for end-users.
- **Implementation**: Streamlit is used for document upload, querying, and displaying responses.
- **Benefit**: It is easy to deploy and integrates seamlessly with Python-based backends.

### **6. Docker for Portability**
- **Purpose**: Enable consistent deployment across environments.
- **Implementation**: A `Dockerfile` encapsulates dependencies and application logic.
- **Benefit**: Simplifies deployment and reduces "it works on my machine" issues.

### **7. GitHub Actions for CI/CD**
- **Purpose**: Automate code quality checks and deployments.
- **Implementation**: A workflow file runs `pytest`, `Bandit`, and Docker build steps on every push or pull request.
- **Benefit**: Ensures code stability and reduces manual intervention during development.

### **8. Use Cases in SDLC**
- **Purpose**: Demonstrate practical applications of RAG in real-world software development workflows.
- **Implementation**: Features like generating user stories, automating test case creation, and validating architectural decisions.
- **Benefit**: Highlights how GenAI can reduce manual effort and improve productivity.

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

## **Business Requirements Document (BRD)**

For detailed business requirements, including project goals, stakeholders, functional and non-functional requirements, and success metrics, see the [Business Requirements Document (BRD)](./docs/brd.md).

### **Executive Summary**
The Conversational RAG Application aims to integrate GenAI into the SDLC to automate and enhance workflows. By enabling document processing, semantic retrieval, and AI-driven response generation, this project empowers business analysts, developers, testers, and architects to:
- Streamline document analysis and requirement generation.
- Reduce manual overhead in testing and development.
- Improve decision-making with AI-assisted insights.

By combining modular architecture, scalable design, and modern DevSecOps practices, this project sets a foundation for scalable, real-world GenAI applications.


## **Use Cases in the SDLC**
- **Business Analysts**: Process BRDs and extract features or user stories.
- **Developers**: Generate boilerplate code from design documents.
- **Testers**: Create test cases or automate test scripts based on requirements.
- **Architects**: Validate design decisions using conversational queries.

## **Contributing**
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.
