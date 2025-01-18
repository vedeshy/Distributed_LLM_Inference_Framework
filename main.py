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

# Set up logging with more detailed metrics
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Metrics:
    def __init__(self):
        self.request_times = []
        self.throughput = []
        self.start_time = time.time()
        self.cpu_usage = []
        
    def log_request(self, duration):
        self.request_times.append(duration)
            
    def calculate_throughput(self):
        elapsed = time.time() - self.start_time
        self.throughput.append(len(self.request_times) / elapsed)
        
    def log_cpu_usage(self):
        self.cpu_usage.append(psutil.cpu_percent())

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize weights
        scale = np.sqrt(2.0 / (d_model + self.d_k))
        self.w_q = np.random.normal(0, scale, (d_model, d_model))
        self.w_k = np.random.normal(0, scale, (d_model, d_model))
        self.w_v = np.random.normal(0, scale, (d_model, d_model))
        self.w_o = np.random.normal(0, scale, (d_model, d_model))
        
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
        output = np.matmul(attention_weights, v)
        
        return output
        
    def forward(self, q, k, v, mask=None):
        q = np.dot(q, self.w_q)
        k = np.dot(k, self.w_k)
        v = np.dot(v, self.w_v)
        
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        scaled_attention = self.scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = scaled_attention.transpose(0, 2, 1, 3).reshape(-1, self.d_model)
        
        return np.dot(scaled_attention, self.w_o)

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Initialize weights
        scale = np.sqrt(2.0 / (d_model + d_ff))
        self.w1 = np.random.normal(0, scale, (d_model, d_ff))
        self.w2 = np.random.normal(0, scale, (d_ff, d_model))
        
    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        
    def forward(self, x):
        return np.dot(self.gelu(np.dot(x, self.w1)), self.w2)

class TransformerModel:
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, num_layers=6):
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.attention_layers = [MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)]
        self.ff_layers = [FeedForward(d_model, d_ff) for _ in range(num_layers)]
        
        # Layer normalization parameters
        self.layer_norms = [(np.ones(d_model), np.zeros(d_model)) for _ in range(num_layers * 2)]
        
    def layer_norm(self, x, g, b, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        return g * (x - mean) / np.sqrt(variance + eps) + b
        
    def forward(self, x, mask=None):
        for i in range(self.num_layers):
            # Multi-head attention
            norm_x = self.layer_norm(x, *self.layer_norms[i*2])
            attention_output = self.attention_layers[i].forward(norm_x, norm_x, norm_x, mask)
            x = x + attention_output
            
            # Feed forward
            norm_x = self.layer_norm(x, *self.layer_norms[i*2+1])
            ff_output = self.ff_layers[i].forward(norm_x)
            x = x + ff_output
            
        return x

class TextPreprocessor:
    def __init__(self, d_model=512, max_sequence_length=128):
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        # Initialize tokenizer (using BERT's tokenizer as an example)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
    def get_positional_encoding(self, seq_length):
        position = np.arange(seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pos_encoding = np.zeros((seq_length, self.d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return pos_encoding
    
    def embed_tokens(self, token_ids):
        # Simple embedding using one-hot encoding and linear projection
        vocab_size = self.tokenizer.vocab_size
        one_hot = np.zeros((len(token_ids), vocab_size))
        for i, token_id in enumerate(token_ids):
            one_hot[i, token_id] = 1
        
        # Project to d_model dimensions using random projection matrix
        projection_matrix = np.random.normal(0, 1/np.sqrt(vocab_size), (vocab_size, self.d_model))
        return np.dot(one_hot, projection_matrix)
    
    def preprocess(self, text):
        # Tokenize
        tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors='np'
        )
        
        # Get token embeddings
        token_embeddings = self.embed_tokens(tokens['input_ids'][0])
        
        # Add positional encoding
        pos_encoding = self.get_positional_encoding(self.max_sequence_length)
        
        # Combine token embeddings and positional encoding
        final_embedding = token_embeddings + pos_encoding
        
        # Apply layer normalization
        mean = np.mean(final_embedding, axis=-1, keepdims=True)
        std = np.std(final_embedding, axis=-1, keepdims=True) + 1e-12
        normalized_embedding = (final_embedding - mean) / std
        
        return normalized_embedding

class SingleDeviceInference:
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size
        self.request_queue = Queue()
        self.metrics = Metrics()
        self.preprocessor = TextPreprocessor(d_model=model.d_model)
        
        # Start worker thread
        self.worker = Thread(target=self._process_batch)
        self.worker.daemon = True
        self.worker.start()
    
    def _prepare_batch(self, inputs):
        processed_inputs = []
        for text in inputs:
            embedding = self.preprocessor.preprocess(text)
            logger.debug(f"Embedding shape after preprocessing: {embedding.shape}")
            averaged_embedding = np.mean(embedding, axis=0)
            logger.debug(f"Embedding shape after averaging: {averaged_embedding.shape}")
            processed_inputs.append(averaged_embedding)
        
        batch = np.stack(processed_inputs)
        logger.debug(f"Final batch shape: {batch.shape}")
        
        if len(batch.shape) == 1:
            batch = batch.reshape(1, -1)
            logger.debug(f"Reshaped batch shape: {batch.shape}")
        
        return batch
    
    def _process_batch(self):
        while True:
            requests = []
            try:
                while len(requests) < self.batch_size:
                    request = self.request_queue.get(timeout=0.1)
                    requests.append(request)
            except queue.Empty:
                pass
            
            if requests:
                self._process_requests(requests)
            time.sleep(0.001)  # Small sleep to prevent CPU spinning
    
    def _process_requests(self, requests):
        start_time = time.time()
        
        try:
            # Batch processing
            input_batch = self._prepare_batch([req['input'] for req in requests])
            
            # Process on CPU using numpy
            outputs = self.model.forward(input_batch)
            
            # Process results
            for req, output in zip(requests, outputs):
                req['callback'](output.tolist())
            
            duration = time.time() - start_time
            self.metrics.log_request(duration)
            self.metrics.log_cpu_usage()
            self.metrics.calculate_throughput()
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            for req in requests:
                req['callback'](None)

class InferenceServer(BaseHTTPRequestHandler):
    inference_engine = None  # Class variable to store inference engine
    
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        request_data = json.loads(post_data.decode('utf-8'))
        
        try:
            input_text = request_data['text']
            
            # Create response callback
            result_ready = threading.Event()
            response_data = {}
            
            def callback(result):
                response_data['result'] = result
                result_ready.set()
            
            # Queue request
            self.inference_engine.request_queue.put({
                'input': input_text,
                'callback': callback
            })
            
            # Wait for result
            if result_ready.wait(timeout=30):
                if response_data['result'] is not None:
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = json.dumps({
                        'result': response_data['result'],
                        'metrics': {
                            'latency': self.inference_engine.metrics.request_times[-1],
                            'throughput': self.inference_engine.metrics.throughput[-1],
                            'cpu_usage': self.inference_engine.metrics.cpu_usage[-1]
                        }
                    })
                    self.wfile.write(response.encode('utf-8'))
                else:
                    raise RuntimeError("Processing failed")
            else:
                raise TimeoutError("Request timed out")
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = json.dumps({'error': str(e)})
            self.wfile.write(response.encode('utf-8'))

def main():
    try:
        logger.info("Initializing transformer model...")
        model = TransformerModel()
        
        logger.info("Setting up inference engine...")
        inference = SingleDeviceInference(model)
        
        # Set the inference engine as a class variable
        InferenceServer.inference_engine = inference
        
        server_address = ('', 8000)
        logger.info("Creating HTTP server...")
        httpd = HTTPServer(server_address, InferenceServer)
        
        logger.info("Starting server on port 8000...")
        httpd.serve_forever()
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

if __name__ == "__main__":
    main()
