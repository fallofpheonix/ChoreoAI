"""
metrics.py — Telemetry and performance tracking.
"""

import time
import torch

class MetricsCollector:
    def __init__(self):
        self.inference_times = []
        self.request_count = 0

    def record_inference(self, start_time):
        latency = time.time() - start_time
        self.inference_times.append(latency)
        self.request_count += 1
        return latency

    def get_summary(self):
        gpu_mem = 0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)
            
        avg_latency = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0
        return {
            "request_count": self.request_count,
            "avg_inference_latency": avg_latency,
            "gpu_memory_allocated_mb": gpu_mem,
            "status": "online"
        }

global_metrics = MetricsCollector()
