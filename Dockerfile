FROM python:3.12.3

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "streamlit_app.py"]