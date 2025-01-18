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
import math

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
        try:
            return {
                'metrics': {
                    'latency': np.mean(self.request_times) if self.request_times else 0,
                    'throughput': self.throughput[-1] if self.throughput else 0,
                    'cpu_usage': np.mean(self.cpu_usage) if self.cpu_usage else 0,
                    'memory_usage': np.mean(self.memory_usage) if self.memory_usage else 0,
                    'total_requests': self.total_requests,
                    'failed_requests': self.failed_requests,
                    'success_rate': (self.total_requests - self.failed_requests) / self.total_requests if self.total_requests > 0 else 0
                }
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {
                'metrics': {
                    'latency': 0,
                    'throughput': 0,
                    'cpu_usage': 0,
                    'memory_usage': 0,
                    'total_requests': self.total_requests,
                    'failed_requests': self.failed_requests,
                    'success_rate': 0
                }
            }

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()  # Initialize the parent class
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Weights will be initialized in TransformerModel.__init__
        self.w_q = None
        self.w_k = None
        self.w_v = None
        self.w_o = None
        
        self.dropout_rate = 0.1
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        
    def forward(self, q, k, v, mask=None):
        # Ensure all inputs are on the same device
        device = q.device
        
        # Linear transformations
        q = torch.matmul(q, self.w_q)
        k = torch.matmul(k, self.w_k)
        v = torch.matmul(v, self.w_v)
        
        # Split heads
        batch_size = q.shape[0]
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = torch.softmax(scores, dim=-1)
        
        if self.dropout_rate > 0:
            attention_weights = self.dropout(attention_weights)
            
        output = torch.matmul(attention_weights, v)
        
        # Combine heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return torch.matmul(output, self.w_o), attention_weights

class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Initialize weights with better scaling
        scale = np.sqrt(2.0 / (d_model + d_ff))
        self.w1 = torch.nn.Parameter(torch.randn(d_model, d_ff) * scale)
        self.w2 = torch.nn.Parameter(torch.randn(d_ff, d_model) * scale)
        
        # Add dropout
        self.dropout_rate = 0.1
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        
    def gelu(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / np.pi)) * (x + 0.044715 * torch.pow(x, 3))))
        
    def forward(self, x):
        h = torch.matmul(x, self.w1)
        h = self.gelu(h)
        
        if self.dropout_rate > 0:
            h = self.dropout(h)
            
        return torch.matmul(h, self.w2)

class TransformerModel:
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, num_layers=6):
        # Explicitly check and set up MPS device
        if not torch.backends.mps.is_available():
            logger.warning("MPS not available. Check if you're using macOS 12.3+ and have an M1/M2 Mac")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("mps")
            logger.info(f"Using MPS device for GPU acceleration: {self.device}")

        self.d_model = d_model
        self.num_layers = num_layers
        
        # Convert all model parameters to torch tensors and move to GPU
        self.to_gpu = lambda x: torch.tensor(x, dtype=torch.float32, device=self.device)
        
        # Initialize attention and feed forward layers first
        self.attention_layers = [MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)]
        self.ff_layers = [FeedForward(d_model, d_ff) for _ in range(num_layers)]
        
        # Move layers to GPU
        for i in range(num_layers):
            self.attention_layers[i] = self.attention_layers[i].to(self.device)
            self.ff_layers[i] = self.ff_layers[i].to(self.device)
        
        # Initialize weights directly on GPU
        scale = np.sqrt(2.0 / (d_model + d_model // num_heads))
        for i in range(num_layers):
            self.attention_layers[i].w_q = self.to_gpu(np.random.normal(0, scale, (d_model, d_model)))
            self.attention_layers[i].w_k = self.to_gpu(np.random.normal(0, scale, (d_model, d_model)))
            self.attention_layers[i].w_v = self.to_gpu(np.random.normal(0, scale, (d_model, d_model)))
            self.attention_layers[i].w_o = self.to_gpu(np.random.normal(0, scale, (d_model, d_model)))
            
        # Initialize layer normalization parameters
        self.layer_norms = [(
            torch.ones(d_model, device=self.device),
            torch.zeros(d_model, device=self.device)
        ) for _ in range(num_layers * 2)]
            
        # Warm up the GPU
        self._warmup()
        
        # Create model directory if it doesn't exist
        self.model_dir = "saved_models"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
    def _warmup(self):
        """Warm up the GPU with a dummy forward pass"""
        logger.info("Warming up GPU...")
        # Create dummy input with correct dimensions [batch_size, seq_length, d_model]
        dummy_input = torch.randn(1, self.d_model, self.d_model, device=self.device)
        # Create dummy mask with correct dimensions [batch_size, seq_length, seq_length]
        dummy_mask = torch.ones(1, self.d_model, self.d_model, device=self.device)
        
        with torch.no_grad():
            for _ in range(3):
                self.forward(dummy_input, dummy_mask)
        logger.info("GPU warmup complete")
        
    def layer_norm(self, x, g, b, eps=1e-5):
        # Ensure input is a torch tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        # Use PyTorch's mean and var operations
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Ensure g and b are on the same device as x
        g = g.to(self.device)
        b = b.to(self.device)
        
        return g * (x - mean) / torch.sqrt(variance + eps) + b
        
    def forward(self, x, mask=None, return_attention=False):
        # Ensure input is on GPU and has correct shape
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        
        # Ensure x has shape [batch_size, seq_length, d_model]
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing
            
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.float32)
            mask = mask.to(self.device)
            # Ensure mask has shape [batch_size, seq_length, seq_length]
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
                
        with torch.no_grad():
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
            
            # Move results back to CPU only at the end
            x = x.cpu().numpy()
            attention_maps = [att.cpu().numpy() for att in attention_maps]
            
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
        # Verify GPU availability
        if torch.backends.mps.is_available():
            logger.info("MPS (Metal Performance Shaders) is available for GPU acceleration")
            # Test GPU with a small tensor
            test_tensor = torch.randn(2, 2, device="mps")
            logger.info(f"Test tensor created on GPU: {test_tensor.device}")
        else:
            logger.warning("MPS not available, falling back to CPU. Requirements:")
            logger.warning("- macOS 12.3 or later")
            logger.warning("- M1/M2 Mac or compatible GPU")
            logger.warning("- PyTorch installed with MPS support")
            
        logger.info("Initializing transformer model...")
        model = TransformerModel()
        
        # Log device placement
        logger.info(f"Model initialized on device: {model.device}")
        
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
