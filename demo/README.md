# MiRAGE Demo

Interactive demo for the MiRAGE Multimodal Multihop QA Dataset Generator.

This demo consists of two components:
1. **Gradio UI** - Frontend interface for Hugging Face Spaces
2. **FastAPI Service** - Backend server that runs the MiRAGE pipeline

## Architecture

```
┌─────────────────────────┐         ┌─────────────────────────┐
│   Hugging Face Spaces   │         │    Your Server/Cloud    │
│                         │         │                         │
│   ┌─────────────────┐   │  HTTP   │   ┌─────────────────┐   │
│   │   Gradio UI     │──────────────>  │  FastAPI Service │   │
│   │  (gradio_app.py)│   │ Request │   │ (fastapi_service │   │
│   └─────────────────┘   │         │   │     .py)         │   │
│                         │         │   └────────┬────────┘   │
│   - File upload         │         │            │            │
│   - API key input       │         │            v            │
│   - Model selection     │         │   ┌─────────────────┐   │
│   - Results display     │         │   │  MiRAGE Pipeline │   │
│                         │         │   │  (src/mirage/)   │   │
└─────────────────────────┘         │   └─────────────────┘   │
                                    └─────────────────────────┘
```

## Demo Limits

To control resource consumption (this is a free demo):

| Parameter | Limit | Description |
|-----------|-------|-------------|
| Max Pages | 20 | Maximum total pages across all uploaded documents |
| Max QA Pairs | 50 | Maximum question-answer pairs to generate |
| Max Depth | 2 | Multi-hop retrieval depth (fixed) |
| Max Breadth | 5 | Search queries per iteration (fixed) |
| Max File Size | 50 MB | Per-file size limit |
| Max Workers | 2 | Parallel processing workers |

## Quick Start

### Option 1: Local Development (Both UI and Backend on same machine)

```bash
# 1. Install dependencies
cd demo
pip install -r requirements.txt

# 2. Install MiRAGE (from parent directory)
pip install -e ..

# 3. Start the FastAPI backend (Terminal 1)
python fastapi_service.py
# Backend runs at http://localhost:8000

# 4. Start the Gradio UI (Terminal 2)
python gradio_app.py
# UI runs at http://localhost:7860

# 5. Open http://localhost:7860 in your browser
```

### Option 2: Separate Deployment (Recommended for Production)

#### Step 1: Deploy FastAPI Backend

Deploy the FastAPI service on a server with GPU (recommended) or CPU:

```bash
# On your server
cd demo

# Install all dependencies
pip install -r requirements.txt
pip install mirage-benchmark[all]

# Set environment variables (optional)
export MIRAGE_DEMO_MAX_PAGES=20
export MIRAGE_DEMO_MAX_QA_PAIRS=50
export PORT=8000
export HOST=0.0.0.0

# Start the service
uvicorn fastapi_service:app --host 0.0.0.0 --port 8000

# For production, use gunicorn:
# pip install gunicorn
# gunicorn fastapi_service:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

#### Step 2: Deploy Gradio UI to Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space)
   - Select "Gradio" as the SDK
   - Choose hardware (CPU is sufficient for UI)

2. Upload the following files to your Space:
   - `gradio_app.py` (rename to `app.py`)
   - `requirements.txt` (Gradio dependencies only)

3. Set the `FASTAPI_URL` secret in your Space settings:
   - Go to Settings > Repository secrets
   - Add: `FASTAPI_URL` = `https://your-backend-server.com`

4. Your Space will automatically build and deploy

## Files

| File | Description |
|------|-------------|
| `gradio_app.py` | Gradio UI frontend for Hugging Face Spaces |
| `fastapi_service.py` | FastAPI backend that runs MiRAGE pipeline |
| `requirements.txt` | Python dependencies for the demo |
| `README.md` | This documentation file |

## API Endpoints

The FastAPI service exposes the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check and service info |
| `/health` | GET | Detailed health status |
| `/process` | POST | Submit documents for processing |
| `/status/{job_id}` | GET | Check job status and progress |
| `/result/{job_id}` | GET | Get processing results |
| `/job/{job_id}` | DELETE | Delete job and cleanup files |

### Example API Usage

```python
import requests

# Submit documents
files = [('files', ('doc.pdf', open('doc.pdf', 'rb')))]
data = {
    'api_key': 'your-gemini-api-key',
    'backend': 'gemini',
    'num_qa_pairs': 50
}
response = requests.post('http://localhost:8000/process', files=files, data=data)
job_id = response.json()['job_id']

# Poll for status
while True:
    status = requests.get(f'http://localhost:8000/status/{job_id}').json()
    if status['status'] in ['completed', 'failed']:
        break
    time.sleep(2)

# Get results
result = requests.get(f'http://localhost:8000/result/{job_id}').json()
print(f"Generated {len(result['qa_pairs'])} QA pairs")
```

## Environment Variables

### FastAPI Service

| Variable | Default | Description |
|----------|---------|-------------|
| `MIRAGE_DEMO_MAX_PAGES` | 20 | Maximum pages allowed |
| `MIRAGE_DEMO_MAX_QA_PAIRS` | 50 | Maximum QA pairs to generate |
| `PORT` | 8000 | Server port |
| `HOST` | 0.0.0.0 | Server host |

### Gradio UI

| Variable | Default | Description |
|----------|---------|-------------|
| `FASTAPI_URL` | http://localhost:8000 | URL of FastAPI backend |

## Hugging Face Spaces Deployment

### Gradio Space `app.py`

Rename `gradio_app.py` to `app.py` for Hugging Face Spaces.

### Gradio Space `requirements.txt`

Create a minimal requirements file for the Gradio Space:

```txt
gradio>=4.0.0
requests>=2.31.0
```

### Space Secrets

Add the following secret in your Space settings:
- `FASTAPI_URL`: URL of your backend server (e.g., `https://api.yourserver.com`)

## Docker Deployment (Optional)

### FastAPI Backend Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY demo/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install MiRAGE
RUN pip install mirage-benchmark[all]

# Copy application
COPY demo/fastapi_service.py .
COPY src/ ./src/

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "fastapi_service:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Run

```bash
# Build
docker build -t mirage-demo-backend -f Dockerfile .

# Run
docker run -p 8000:8000 mirage-demo-backend
```

## Troubleshooting

### "Cannot connect to backend"

1. Verify the FastAPI service is running: `curl http://localhost:8000/health`
2. Check the `FASTAPI_URL` environment variable in Gradio
3. Ensure firewall allows connections on port 8000

### "Processing timed out"

1. Processing large documents takes time; try smaller files
2. Check backend logs for errors
3. Verify API key is correct

### "API key required"

1. For Gemini/OpenAI backends, an API key is required
2. For Ollama, ensure the local Ollama server is running and accessible from the backend

### "Total pages exceeds limit"

1. Upload fewer documents or use smaller PDFs
2. Demo limit is 20 pages total

## Security Notes

1. **API Keys**: API keys are sent securely to the backend but not stored permanently. In production, consider using environment variables or secret managers.

2. **CORS**: The default CORS configuration allows all origins (`*`). For production, restrict to your Gradio Space URL.

3. **File Uploads**: Uploaded files are stored temporarily and cleaned up after processing. Implement additional cleanup cron jobs for production.

4. **Rate Limiting**: Consider adding rate limiting for production deployments to prevent abuse.

## Support

- GitHub Issues: https://github.com/ChandanKSahu/MiRAGE/issues
- Documentation: https://github.com/ChandanKSahu/MiRAGE#readme

## License

Apache License 2.0 - see [LICENSE](../LICENSE)
