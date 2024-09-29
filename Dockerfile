FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

# Copy the application and models
COPY ./app /app
COPY ./models /app

# Define the command to run the application
CMD ["python", "flask_app.py"]