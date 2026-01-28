# 1. Base Image (python:3.11-slim for less size)
FROM python:3.11-slim

# 2. Set Work Directory inside Container
WORKDIR /app

# 3. Copy Code (From current folder to Container at /app)
COPY . /app

# 4. Install Dependencies
RUN apt-get update && apt-get install -y git && \
    pip install --no-cache-dir -r requirements.txt

# 5. Set Port No. (FastAPI works on 8000)
EXPOSE 8000

# 6. Command to start server
CMD ["python", "app.py"]