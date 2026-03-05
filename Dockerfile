FROM python:3.10-slim

WORKDIR /app

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Copy project
COPY . .

# Install python packages
RUN pip install --no-cache-dir -r requirements.txt

# Run server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]