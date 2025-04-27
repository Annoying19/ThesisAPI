# Use an official lightweight Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Google Cloud expects
ENV PORT 8080

# Run the app with gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "--timeout", "300", "main:app"]

