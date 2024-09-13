import time
import torch
import numpy as np
import psutil

class Profiler:
    def __init__(self):
        self.logger = None
        self.inference_times = []
        self.epoch_times = []
        self.peak_ram = 0
        self.peak_vram = 0

    def set_logger(self, logger):
        self.logger = logger

    def measure_time(self, func, *args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, end - start

    def update_peak_memory(self):
        ram = psutil.virtual_memory().used / 1e9  # GB
        vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0  # GB
        self.peak_ram = max(self.peak_ram, ram)
        self.peak_vram = max(self.peak_vram, vram)

    def update_epoch_time(self, time):
        self.epoch_times.append(time)

    def update_inference_time(self, time):
        self.inference_times.append(time)

    def reset_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def report(self, is_training=True):
        if is_training:
            avg_time = np.mean(self.epoch_times)
            std_time = np.std(self.epoch_times)
            time_msg = f"Average training time per epoch: {avg_time:.2f} ± {std_time:.2f} seconds"
        else:
            avg_time = np.mean(self.inference_times)
            std_time = np.std(self.inference_times)
            time_msg = f"Average inference time per patient: {avg_time:.4f} ± {std_time:.4f} seconds"

        self.logger.info(time_msg)
        self.logger.info(f"Peak RAM usage: {self.peak_ram:.2f} GB")
        self.logger.info(f"Peak VRAM usage: {self.peak_vram:.2f} GB")


# Create global instances
train_profiler = Profiler()
test_profiler = Profiler()