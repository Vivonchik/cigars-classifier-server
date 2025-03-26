FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
CMD ["sh", "-c", "streamlit run online_server.py --server.port=$PORT --server.enableCORS=false"]
