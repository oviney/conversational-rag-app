FROM python:3.12.3

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the PYTHONPATH environment variable to ensure proper imports
ENV PYTHONPATH="/conversational-rag-app"

# Expose the port the app runs on
EXPOSE 8501

# Default command to run the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py"]
