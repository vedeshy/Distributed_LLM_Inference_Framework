import numpy as np
import logging
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from queue import Queue, Empty
from threading import Thread
import psutil
import threading
import queue
from transformers import AutoTokenizer
import torch
import os
import sys
from datetime import datetime
import pickle
import signal

# Set up logging with more detailed metrics and file output
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'server_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class Metrics:
    def __init__(self):
        self.request_times = []
        self.throughput = []
        self.start_time = time.time()
        self.cpu_usage = []
        self.memory_usage = []
        self.total_requests = 0
        self.failed_requests = 0
        
    def log_request(self, duration):
        self.request_times.append(duration)
        self.total_requests += 1
            
    def calculate_throughput(self):
        elapsed = time.time() - self.start_time
        self.throughput.append(len(self.request_times) / elapsed)
        
    def log_cpu_usage(self):
        self.cpu_usage.append(psutil.cpu_percent())
        self.memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB
        
    def log_failure(self):
        self.failed_requests += 1
        
    def get_statistics(self):
        if not self.request_times:
            return {}
        return {
            'avg_latency': np.mean(self.request_times),
            'p95_latency': np.percentile(self.request_times, 95),
            'p99_latency': np.percentile(self.request_times, 99),
            'min_latency': np.min(self.request_times),
            'max_latency': np.max(self.request_times),
            'current_throughput': self.throughput[-1] if self.throughput else 0,
            'avg_cpu_usage': np.mean(self.cpu_usage),
            'avg_memory_usage_mb': np.mean(self.memory_usage),
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (self.total_requests - self.failed_requests) / self.total_requests if self.total_requests > 0 else 0
        }

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize weights with better scaling
        scale = np.sqrt(2.0 / (d_model + self.d_k))
        self.w_q = np.random.normal(0, scale, (d_model, d_model))
        self.w_k = np.random.normal(0, scale, (d_model, d_model))
        self.w_v = np.random.normal(0, scale, (d_model, d_model))
        self.w_o = np.random.normal(0, scale, (d_model, d_model))
        
        # Add dropout
        self.dropout_rate = 0.1
        
    def split_heads(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        matmul_qk = np.matmul(q, k.transpose(0, 1, 3, 2))
        scale = np.sqrt(self.d_k)
        scaled_attention_logits = matmul_qk / scale
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
            
        attention_weights = np.exp(scaled_attention_logits) / np.sum(np.exp(scaled_attention_logits), axis=-1, keepdims=True)
        
        # Apply dropout
        if self.dropout_rate > 0:
            dropout_mask = np.random.binomial(1, 1-self.dropout_rate, attention_weights.shape)
            attention_weights *= dropout_mask / (1 - self.dropout_rate)
            
        output = np.matmul(attention_weights, v)
        
        return output, attention_weights
        
    def forward(self, q, k, v, mask=None):
        q = np.dot(q, self.w_q)
        k = np.dot(k, self.w_k)
        v = np.dot(v, self.w_v)
        
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = scaled_attention.transpose(0, 2, 1, 3).reshape(-1, self.d_model)
        
        return np.dot(scaled_attention, self.w_o), attention_weights

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Initialize weights with better scaling
        scale = np.sqrt(2.0 / (d_model + d_ff))
        self.w1 = np.random.normal(0, scale, (d_model, d_ff))
        self.w2 = np.random.normal(0, scale, (d_ff, d_model))
        
        # Add dropout
        self.dropout_rate = 0.1
        
    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        
    def forward(self, x):
        h = np.dot(x, self.w1)
        h = self.gelu(h)
        
        if self.dropout_rate > 0:
            dropout_mask = np.random.binomial(1, 1-self.dropout_rate, h.shape)
            h *= dropout_mask / (1 - self.dropout_rate)
            
        return np.dot(h, self.w2)

class TransformerModel:
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, num_layers=6):
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.attention_layers = [MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)]
        self.ff_layers = [FeedForward(d_model, d_ff) for _ in range(num_layers)]
        
        # Layer normalization parameters
        self.layer_norms = [(np.ones(d_model), np.zeros(d_model)) for _ in range(num_layers * 2)]
        
        # Save/load methods
        self.model_dir = "saved_models"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
    def layer_norm(self, x, g, b, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        return g * (x - mean) / np.sqrt(variance + eps) + b
        
    def forward(self, x, mask=None, return_attention=False):
        attention_maps = []
        
        for i in range(self.num_layers):
            # Multi-head attention
            norm_x = self.layer_norm(x, *self.layer_norms[i*2])
            attention_output, attention_weights = self.attention_layers[i].forward(norm_x, norm_x, norm_x, mask)
            attention_maps.append(attention_weights)
            x = x + attention_output
            
            # Feed forward
            norm_x = self.layer_norm(x, *self.layer_norms[i*2+1])
            ff_output = self.ff_layers[i].forward(norm_x)
            x = x + ff_output
            
        if return_attention:
            return x, attention_maps
        return x
        
    def save(self, filename):
        path = os.path.join(self.model_dir, filename)
        with open(path, 'wb') as f:
            pickle.dump({
                'd_model': self.d_model,
                'num_layers': self.num_layers,
                'attention_layers': self.attention_layers,
                'ff_layers': self.ff_layers,
                'layer_norms': self.layer_norms
            }, f)
        logger.info(f"Model saved to {path}")
        
    def load(self, filename):
        path = os.path.join(self.model_dir, filename)
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.d_model = data['d_model']
            self.num_layers = data['num_layers']
            self.attention_layers = data['attention_layers']
            self.ff_layers = data['ff_layers']
            self.layer_norms = data['layer_norms']
        logger.info(f"Model loaded from {path}")

class TextPreprocessor:
    def __init__(self, d_model=512, max_sequence_length=128):
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Cache for positional encodings
        self.pos_encoding_cache = {}
        
    def get_positional_encoding(self, seq_length):
        if seq_length in self.pos_encoding_cache:
            return self.pos_encoding_cache[seq_length]
            
        position = np.arange(seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pos_encoding = np.zeros((seq_length, self.d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        self.pos_encoding_cache[seq_length] = pos_encoding
        return pos_encoding
    
    def embed_tokens(self, token_ids):
        vocab_size = self.tokenizer.vocab_size
        one_hot = np.zeros((len(token_ids), vocab_size))
        for i, token_id in enumerate(token_ids):
            one_hot[i, token_id] = 1
        
        if not hasattr(self, 'projection_matrix'):
            self.projection_matrix = np.random.normal(0, 1/np.sqrt(vocab_size), (vocab_size, self.d_model))
            
        return np.dot(one_hot, self.projection_matrix)
    
    def preprocess(self, text, return_attention_mask=False):
        tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors='np'
        )
        
        token_embeddings = self.embed_tokens(tokens['input_ids'][0])
        pos_encoding = self.get_positional_encoding(self.max_sequence_length)
        final_embedding = token_embeddings + pos_encoding
        
        mean = np.mean(final_embedding, axis=-1, keepdims=True)
        std = np.std(final_embedding, axis=-1, keepdims=True) + 1e-12
        normalized_embedding = (final_embedding - mean) / std
        
        if return_attention_mask:
            return normalized_embedding, tokens['attention_mask']
        return normalized_embedding

class SingleDeviceInference:
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size
        self.request_queue = Queue()
        self.metrics = Metrics()
        self.preprocessor = TextPreprocessor(d_model=model.d_model)
        
        # Initialize the running flag before starting the worker thread
        self.running = True
        
        # Start worker thread
        self.worker = Thread(target=self._process_batch)
        self.worker.daemon = True
        self.worker.start()
        
        # Add signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        logger.info("Received shutdown signal, cleaning up...")
        self.running = False
        # Save final metrics
        self._save_metrics()
        sys.exit(0)
        
    def _save_metrics(self):
        stats = self.metrics.get_statistics()
        with open(os.path.join(log_dir, 'final_metrics.json'), 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _prepare_batch(self, inputs):
        processed_inputs = []
        attention_masks = []
        for text in inputs:
            embedding, mask = self.preprocessor.preprocess(text, return_attention_mask=True)
            logger.debug(f"Embedding shape after preprocessing: {embedding.shape}")
            processed_inputs.append(embedding)
            attention_masks.append(mask)
        
        batch = np.stack(processed_inputs)
        masks = np.stack(attention_masks)
        logger.debug(f"Final batch shape: {batch.shape}")
        
        if len(batch.shape) == 1:
            batch = batch.reshape(1, -1)
            logger.debug(f"Reshaped batch shape: {batch.shape}")
        
        return batch, masks
    
    def _process_batch(self):
        while self.running:
            requests = []
            try:
                while len(requests) < self.batch_size:
                    request = self.request_queue.get(timeout=0.1)
                    requests.append(request)
            except queue.Empty:
                pass
            
            if requests:
                self._process_requests(requests)
            time.sleep(0.001)
    
    def _process_requests(self, requests):
        start_time = time.time()
        
        try:
            input_batch, attention_masks = self._prepare_batch([req['input'] for req in requests])
            
            outputs, attention_maps = self.model.forward(input_batch, mask=attention_masks, return_attention=True)
            
            for req, output, attention in zip(requests, outputs, attention_maps[-1]):
                req['callback']({
                    'output': output.tolist(),
                    'attention': attention.tolist()
                })
            
            duration = time.time() - start_time
            self.metrics.log_request(duration)
            self.metrics.log_cpu_usage()
            self.metrics.calculate_throughput()
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            self.metrics.log_failure()
            for req in requests:
                req['callback'](None)

class InferenceServer(BaseHTTPRequestHandler):
    inference_engine = None
    
    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            stats = self.inference_engine.metrics.get_statistics()
            self.wfile.write(json.dumps(stats).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        request_data = json.loads(post_data.decode('utf-8'))
        
        try:
            input_text = request_data['text']
            
            result_ready = threading.Event()
            response_data = {}
            
            def callback(result):
                response_data['result'] = result
                result_ready.set()
            
            self.inference_engine.request_queue.put({
                'input': input_text,
                'callback': callback
            })
            
            if result_ready.wait(timeout=30):
                if response_data['result'] is not None:
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = json.dumps({
                        # 'result': response_data['result'],
                        'metrics': {
                            'latency': self.inference_engine.metrics.request_times[-1],
                            'throughput': self.inference_engine.metrics.throughput[-1],
                            'cpu_usage': self.inference_engine.metrics.cpu_usage[-1],
                            'memory_usage': self.inference_engine.metrics.memory_usage[-1]
                        }
                    })
                    self.wfile.write(response.encode('utf-8'))
                else:
                    raise RuntimeError("Processing failed")
            else:
                raise TimeoutError("Request timed out")
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            self.inference_engine.metrics.log_failure()
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = json.dumps({'error': str(e)})
            self.wfile.write(response.encode('utf-8'))

def main():
    try:
        logger.info("Initializing transformer model...")
        model = TransformerModel()
        
        # Try to load saved model if exists
        if os.path.exists(os.path.join(model.model_dir, 'latest.pkl')):
            try:
                model.load('latest.pkl')
            except Exception as e:
                logger.warning(f"Could not load saved model: {e}")
        
        logger.info("Setting up inference engine...")
        inference = SingleDeviceInference(model)
        
        InferenceServer.inference_engine = inference
        
        # Changed from ('', 8000) to ('0.0.0.0', 8000) to explicitly bind to all interfaces
        server_address = ('0.0.0.0', 8000)
        logger.info("Creating HTTP server...")
        httpd = HTTPServer(server_address, InferenceServer)
        
        logger.info(f"Starting server on {server_address[0]}:{server_address[1]}...")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down server...")
            httpd.server_close()
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

if __name__ == "__main__":
    main()
