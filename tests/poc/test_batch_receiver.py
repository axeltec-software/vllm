#!/usr/bin/env python3

import time
import requests
import threading
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from typing import List, Optional

MLNODE_URL = "http://localhost:5000"
API_PREFIX = "/api/v1"
BATCH_RECEIVER_PORT = 9999
BATCH_RECEIVER_LOCAL_URL = f"http://localhost:{BATCH_RECEIVER_PORT}"

MODEL_NAME = "RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic"

POW_CONFIG = {
    "block_hash": "TEST_BLOCK",
    "block_height": 100,
    "public_key": "test_pub_keys",
    "node_id": 0,
    "node_count": 1,
    "params": {
        "model": MODEL_NAME,
        "seq_len": 1024,
        "k_dim": 12,
    },
}

_proof_batches: List[dict] = []
_server_instance: Optional[HTTPServer] = None


class BatchReceiverHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass
    
    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _count_nonces(self, batch: dict) -> int:
        if "artifacts" in batch:
            return len(batch["artifacts"])
        elif "nonces" in batch:
            return len(batch["nonces"])
        return 0
    
    def do_GET(self):
        if self.path == '/health':
            self._send_json({"status": "OK"})
        elif self.path == '/stats':
            total_nonces = sum(self._count_nonces(b) for b in _proof_batches)
            batch_count = len(_proof_batches)
            batch_sizes = [self._count_nonces(b) for b in _proof_batches]
            avg_batch_size = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0
            self._send_json({
                "total_nonces": total_nonces,
                "batch_count": batch_count,
                "batch_sizes": batch_sizes,
                "avg_batch_size": avg_batch_size,
            })
        elif self.path == '/batches':
            self._send_json({"batches": _proof_batches})
        else:
            self._send_json({"error": "Not found"}, 404)
    
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode() if content_length > 0 else '{}'
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
            return
        
        if self.path == '/generated':
            _proof_batches.append(data)
            nonce_count = self._count_nonces(data)
            print(f"Received batch #{len(_proof_batches)}: {nonce_count} nonces")
            self._send_json({"message": "OK", "batch_count": len(_proof_batches)})
        elif self.path == '/clear':
            _proof_batches.clear()
            self._send_json({"message": "Cleared"})
        else:
            self._send_json({"error": "Not found"}, 404)


def run_server(port: int):
    global _server_instance
    _server_instance = HTTPServer(('0.0.0.0', port), BatchReceiverHandler)
    _server_instance.serve_forever()


def start_batch_receiver() -> threading.Thread:
    thread = threading.Thread(target=run_server, args=(BATCH_RECEIVER_PORT,), daemon=True)
    thread.start()
    return thread


def stop_batch_receiver():
    global _server_instance
    if _server_instance:
        _server_instance.shutdown()


def wait_for_batch_receiver_ready(timeout_s: int = 10) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout_s:
        try:
            response = requests.get(f"{BATCH_RECEIVER_LOCAL_URL}/health", timeout=2)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)
    return False


def get_batch_receiver_stats() -> dict:
    response = requests.get(f"{BATCH_RECEIVER_LOCAL_URL}/stats", timeout=10)
    response.raise_for_status()
    return response.json()


def clear_batch_receiver():
    response = requests.post(f"{BATCH_RECEIVER_LOCAL_URL}/clear", timeout=10)
    response.raise_for_status()


def init_generate(batch_size: int) -> dict:
    payload = {
        **POW_CONFIG,
        "batch_size": batch_size,
        "url": BATCH_RECEIVER_LOCAL_URL,
    }
    
    response = requests.post(
        f"{MLNODE_URL}{API_PREFIX}/pow/init/generate",
        json=payload, timeout=60
    )
    response.raise_for_status()
    return response.json()

def get_collected_batches() -> List[dict]:
    response = requests.get(f"{BATCH_RECEIVER_LOCAL_URL}/batches", timeout=10)
    response.raise_for_status()
    return response.json().get("batches", [])

def stop_generation() -> dict:
    response = requests.post(
        f"{MLNODE_URL}{API_PREFIX}/pow/stop",
        json={}, timeout=60
    )
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="Test PoC Batch Receiver")
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--interval", type=int, default=5)
    args = parser.parse_args()
    
    receiver_thread = None
    try:
        print("Starting batch receiver...")
        receiver_thread = start_batch_receiver()
        if not wait_for_batch_receiver_ready():
            print("FAILED: Could not start batch receiver")
            return
        clear_batch_receiver()
        print(f"Starting PoC generation (batch_size={args.batch_size})...")
        try:
            result = init_generate(batch_size=args.batch_size)
            print(f"Response: {result.get('message', result.get('status', 'OK'))}")
        except requests.exceptions.RequestException as e:
            print(f"FAILED: Could not start generation: {e}")
            return
        
        start_time = time.time()
        last_report = 0
        while True:
            elapsed = time.time() - start_time
            if elapsed >= args.duration:
                break
            
            if int(elapsed) // args.interval > last_report:
                last_report = int(elapsed) // args.interval
                try:
                    stats = get_batch_receiver_stats()
                    batches = get_collected_batches()
                    print(f"[{int(elapsed):3d}s]")
                    print(f"Nonces: {stats['total_nonces']:4d}")
                    print(batches)
                except Exception as e:
                    print(f"[{int(elapsed):3d}s] ERROR: {e}")
    
            time.sleep(1)
        print("\nStopping generation...")
        try:
            stop_generation()
        except Exception as e:
            print(f"WARNING: Error stopping generation: {e}")

        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        try:
            final_stats = get_batch_receiver_stats()
            
            print(f"Total batches received: {final_stats['batch_count']}")
            print(f"Total nonces: {final_stats['total_nonces']}")
            
            if final_stats['total_nonces'] == 0:
                print("\nWARNING: No nonces received")
        except Exception as e:
            print(f"\nERROR: Could not get final stats: {e}")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        try:
            stop_generation()
        except Exception:
            pass
    finally:
        if receiver_thread:
            print("\nStopping batch receiver...")
            stop_batch_receiver()
            receiver_thread.join(timeout=2)


if __name__ == "__main__":
    exit(main())
