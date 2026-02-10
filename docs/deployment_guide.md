# Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Agentic RAG System in various environments, from local development to production-scale deployments.

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.9 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: 50GB+ available space for documents and embeddings
- **Network**: Internet access for downloading models and dependencies

### Software Dependencies

- **Docker**: For containerized deployment
- **Docker Compose**: For multi-container applications
- **NVIDIA GPU** (optional): For accelerated embeddings and LLM inference

## Local Development Setup

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd agentic-rag-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```env
# LLM Configuration
GOOGLE_API_KEY=your_gemini_api_key_here
OLLAMA_BASE_URL=http://localhost:11434

# Milvus Configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Application Settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K=5
MAX_CHUNK_SIZE=1000

# Optional: Enable debug mode
DEBUG=true
LOG_LEVEL=DEBUG
```

### 3. Start Milvus

```bash
# Start Milvus with Docker Compose
docker-compose -f docker/docker-compose.yml up -d

# Verify Milvus is running
docker ps | grep milvus
```

### 4. Start Ollama (Optional)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve &

# Pull a model (optional)
ollama pull llama3.2
```

### 5. Run the Application

```bash
# Start the Streamlit application
streamlit run app.py

# Open browser to http://localhost:8501
```

## Docker Deployment

### 1. Build Docker Image

```bash
# Build the application image
docker build -t agentic-rag-system .

# Build with specific platform (if needed)
docker build --platform linux/amd64 -t agentic-rag-system .
```

### 2. Docker Compose Deployment

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  # Milvus standalone
  milvus-standalone:
    image: milvusdb/milvus:v2.4.0
    container_name: milvus-standalone
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ./milvus:/var/lib/milvus
    ports:
      - "19530:19530"
    networks:
      - rag-network

  # Etcd
  etcd:
    image: quay.io/coreos/etcd:v3.5.18
    container_name: milvus-etcd
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ./etcd:/etcd
    command: etcd -advertise-client-urls=http://etcd:2379 -listen-client-urls=http://0.0.0.0:2379 --data-dir /etcd
    networks:
      - rag-network

  # MinIO
  minio:
    image: minio/minio:RELEASE.2024-03-14T18-03-29Z
    container_name: milvus-minio
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - ./minio:/data
    command: minio server /data --console-address ":9001"
    networks:
      - rag-network

  # Redis (optional, for caching)
  redis:
    image: redis:7-alpine
    container_name: rag-redis
    ports:
      - "6379:6379"
    volumes:
      - ./redis:/data
    command: redis-server --appendonly yes
    networks:
      - rag-network

  # Ollama (optional)
  ollama:
    image: ollama/ollama:latest
    container_name: rag-ollama
    ports:
      - "11434:11434"
    volumes:
      - ./ollama:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
    networks:
      - rag-network

  # Application
  app:
    build: .
    container_name: agentic-rag-app
    ports:
      - "8501:8501"
    environment:
      - MILVUS_HOST=milvus-standalone
      - MILVUS_PORT=19530
      - OLLAMA_BASE_URL=http://ollama:11434
      - DEBUG=false
    volumes:
      - ./uploads:/app/uploads
      - ./temp:/app/temp
    depends_on:
      - milvus-standalone
      - ollama
    networks:
      - rag-network

networks:
  rag-network:
    driver: bridge
```

### 3. Deploy with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

## Kubernetes Deployment

### 1. Prerequisites

- Kubernetes cluster (v1.20+)
- kubectl configured
- Helm 3.x

### 2. Install Milvus with Helm

```bash
# Add Milvus Helm repository
helm repo add milvus https://milvus-io.github.io/milvus-helm
helm repo update

# Install Milvus
helm install my-milvus milvus/milvus \
  --set cluster.enabled=false \
  --set minio.mode=standalone \
  --set etcd.replicaCount=1 \
  --set pulsar.enabled=false \
  --set dependencies.minio.enabled=true \
  --set dependencies.etcd.enabled=true
```

### 3. Deploy Application

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentic-rag-app
  labels:
    app: agentic-rag
spec:
  replicas: 2
  selector:
    matchLabels:
      app: agentic-rag
  template:
    metadata:
      labels:
        app: agentic-rag
    spec:
      containers:
      - name: app
        image: agentic-rag-system:latest
        ports:
        - containerPort: 8501
        env:
        - name: MILVUS_HOST
          value: "my-milvus-milvus"
        - name: MILVUS_PORT
          value: "19530"
        - name: DEBUG
          value: "false"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        volumeMounts:
        - name: uploads
          mountPath: /app/uploads
        - name: temp
          mountPath: /app/temp
      volumes:
      - name: uploads
        persistentVolumeClaim:
          claimName: rag-uploads-pvc
      - name: temp
        emptyDir: {}

---
apiVersion: v1
kind: Service
metadata:
  name: agentic-rag-service
spec:
  selector:
    app: agentic-rag
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8501
  type: LoadBalancer
```

### 4. Deploy to Kubernetes

```bash
# Apply deployment
kubectl apply -f k8s-deployment.yaml

# Check deployment status
kubectl get pods -l app=agentic-rag
kubectl get services

# Get external IP
kubectl get service agentic-rag-service
```

## Production Deployment

### 1. Environment Setup

#### High Availability Configuration

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  # Milvus cluster setup
  milvus-standalone:
    image: milvusdb/milvus:v2.4.0
    deploy:
      replicas: 2
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - milvus_data:/var/lib/milvus
    networks:
      - rag-network

  # Load balancer for Milvus
  nginx:
    image: nginx:alpine
    ports:
      - "19530:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - milvus-standalone
    networks:
      - rag-network

  # Application with multiple replicas
  app:
    build: .
    deploy:
      replicas: 3
    environment:
      - MILVUS_HOST=nginx
      - MILVUS_PORT=19530
      - DEBUG=false
      - LOG_LEVEL=INFO
    volumes:
      - uploads_data:/app/uploads
    networks:
      - rag-network

volumes:
  milvus_data:
  uploads_data:

networks:
  rag-network:
    driver: bridge
```

#### Nginx Configuration

Create `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream milvus_backend {
        server milvus-standalone:19530;
    }

    server {
        listen 80;
        location / {
            proxy_pass http://milvus_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
```

### 2. Monitoring and Logging

#### Prometheus Configuration

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'agentic-rag'
    static_configs:
      - targets: ['app:8501']
  
  - job_name: 'milvus'
    static_configs:
      - targets: ['milvus-standalone:9091']
```

#### Grafana Dashboard

Import dashboard configuration for monitoring:
- Application metrics (response time, error rate)
- Milvus metrics (query latency, storage usage)
- System metrics (CPU, memory, disk)

### 3. Security Configuration

#### SSL/TLS Setup

```yaml
# docker-compose.ssl.yml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx-ssl.conf:/etc/nginx/nginx.conf
      - ./ssl/cert.pem:/etc/ssl/cert.pem
      - ./ssl/key.pem:/etc/ssl/key.pem
    networks:
      - rag-network
```

#### Authentication

Implement authentication using:
- OAuth2/OIDC providers
- API key authentication
- JWT tokens

### 4. Backup and Recovery

#### Milvus Backup

```bash
# Backup Milvus data
docker exec milvus-standalone bash -c "tar -czf /backup/milvus-backup.tar.gz /var/lib/milvus"

# Restore Milvus data
docker exec milvus-standalone bash -c "tar -xzf /backup/milvus-backup.tar.gz -C /"
```

#### Application Data Backup

```bash
# Backup uploads directory
tar -czf uploads-backup.tar.gz uploads/

# Backup configuration
tar -czf config-backup.tar.gz .env docker/
```

## Performance Optimization

### 1. Milvus Optimization

#### Index Configuration

```python
# Optimize for your use case
index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",  # Better than IVF_FLAT for production
    "params": {
        "M": 16,           # Number of connections per layer
        "efConstruction": 200  # Construction parameter
    }
}
```

#### Resource Allocation

```yaml
# docker-compose.performance.yml
version: '3.8'

services:
  milvus-standalone:
    image: milvusdb/milvus:v2.4.0
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
        reservations:
          memory: 4G
          cpus: '2.0'
```

### 2. Application Optimization

#### Caching Strategy

```python
# Redis caching for frequent queries
import redis
from functools import lru_cache

redis_client = redis.Redis(host='redis', port=6379, db=0)

@lru_cache(maxsize=1000)
def cached_query(query: str):
    # Check cache first
    cached_result = redis_client.get(f"query:{query}")
    if cached_result:
        return cached_result
    
    # Process query
    result = process_query(query)
    
    # Cache result
    redis_client.setex(f"query:{query}", 3600, result)  # 1 hour TTL
    return result
```

#### Batch Processing

```python
# Process documents in batches
def process_documents_batch(file_paths: List[str], batch_size: int = 10):
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i + batch_size]
        results = document_pipeline.process_multiple_documents(batch)
        yield results
```

### 3. Monitoring and Alerting

#### Health Checks

```python
# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "milvus_connected": milvus_vector_store.connected,
        "llm_available": check_llm_availability(),
        "timestamp": datetime.now()
    }
```

#### Metrics Collection

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram

QUERY_COUNTER = Counter('rag_queries_total', 'Total number of queries')
QUERY_DURATION = Histogram('rag_query_duration_seconds', 'Query duration')

@QUERY_DURATION.time()
def process_query(query: str):
    QUERY_COUNTER.inc()
    # Process query
    return result
```

## Troubleshooting

### Common Issues

#### Milvus Connection Issues

```bash
# Check Milvus status
docker logs milvus-standalone

# Verify connection
python -c "from src.retrieval.vector_store import milvus_vector_store; print(milvus_vector_store.connect())"
```

#### LLM Connection Issues

```bash
# Test Gemini API
curl -X POST "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key=YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"parts":[{"text":"Hello"}]}]}'

# Test Ollama
curl http://localhost:11434/api/tags
```

#### Performance Issues

```bash
# Check resource usage
docker stats

# Monitor Milvus metrics
curl http://localhost:9091/metrics

# Check application logs
docker logs agentic-rag-app
```

### Debug Mode

Enable debug mode for detailed logging:

```env
DEBUG=true
LOG_LEVEL=DEBUG
```

This deployment guide provides comprehensive instructions for deploying the Agentic RAG System in various environments, ensuring optimal performance and reliability.