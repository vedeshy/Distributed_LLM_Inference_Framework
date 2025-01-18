# Transformer Inference Server

This project is an HTTP server for processing text using a transformer-based model. It includes:

- A transformer model implemented in Python with numpy.
- An inference engine for handling multiple requests in batches.
- Metrics tracking for latency, throughput, CPU usage, and memory usage.
- REST endpoints for submitting inference requests and monitoring metrics.

---

## Features

- **Transformer Model**: Implements multi-head attention and feed-forward layers.
- **Batch Inference**: Processes requests in batches to optimize throughput.
- **Metrics Tracking**: Monitors performance metrics such as latency, CPU, memory usage, and throughput.
- **Graceful Shutdown**: Saves metrics and cleans up resources on shutdown.

---

## Installation

### Clone the Repository
```bash
git clone <repository-url>
cd <repository-folder>
```

### Install Dependencies
Install the required Python libraries:
```bash
pip install numpy transformers psutil
```

### Run the Server
```bash
python <filename>.py
```

---

## API Endpoints

### POST `/request`

**Description**: Processes a text input and returns the transformer model's output along with metrics.

- **Request**:
  - `Content-Type: application/json`
  - **Body**:
    ```json
    {
      "text": "Your input text here"
    }
    ```

- **Response**:
  - `Content-Type: application/json`
  - **Body**:
    ```json
    {
      "result": {
        "output": [[...]],
        "attention": [[...]]
      },
      "metrics": {
        "latency": 0.234,
        "throughput": 12.3,
        "cpu_usage": 23.5,
        "memory_usage": 134.7
      }
    }
    ```

---

### GET `/metrics`

**Description**: Returns the current performance metrics of the server.

- **Request**: None
- **Response**:
  - `Content-Type: application/json`
  - **Body**:
    ```json
    {
      "avg_latency": 0.234,
      "p95_latency": 0.456,
      "p99_latency": 0.678,
      "min_latency": 0.123,
      "max_latency": 0.789,
      "current_throughput": 12.3,
      "avg_cpu_usage": 23.5,
      "avg_memory_usage_mb": 134.7,
      "total_requests": 100,
      "failed_requests": 2,
      "success_rate": 0.98
    }
    ```

---

## Testing the Server

You can test the server using `curl`:

### Submit a Text Inference Request

```bash
curl -X POST http://localhost:8000/request \
-H "Content-Type: application/json" \
-d '{"text": "This is a sample input text."}'
```

**Example Response**:
```json
{
  "result": {
    "output": [[...]],
    "attention": [[...]]
  },
  "metrics": {
    "latency": 0.234,
    "throughput": 12.3,
    "cpu_usage": 23.5,
    "memory_usage": 134.7
  }
}
```

### Retrieve Metrics

```bash
curl -X GET http://localhost:8000/metrics
```

**Example Response**:
```json
{
  "avg_latency": 0.234,
  "p95_latency": 0.456,
  "p99_latency": 0.678,
  "min_latency": 0.123,
  "max_latency": 0.789,
  "current_throughput": 12.3,
  "avg_cpu_usage": 23.5,
  "avg_memory_usage_mb": 134.7,
  "total_requests": 100,
  "failed_requests": 2,
  "success_rate": 0.98
}
```

---

## Logs

Logs are stored in the `logs/` directory with timestamps. Logs include details about request handling, metrics, errors, and server lifecycle events.

---

## Graceful Shutdown

The server captures `SIGINT` and `SIGTERM` signals to:

1. Save final metrics to `logs/final_metrics.json`.
2. Clean up resources and exit safely.

---

## Notes

- Ensure the `logs/` and `saved_models/` directories exist with the correct permissions.
- The server listens on `0.0.0.0:8000` by default and can be modified in the `main()` function.
- The transformer model is lightweight and for demonstration purposes only.

---

## Future Improvements

- Add GPU acceleration for the transformer model.
- Integrate with a production-grade serving framework (e.g., FastAPI, Flask).
- Optimize the batching mechanism for variable-length inputs.
- Add more robust error handling and retry logic.
