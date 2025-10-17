# bench_cpu.py  (solo CPU)
import time, torch

device = 'cpu'
print('device =', device)

# Modelo dummy (Conv2d) y entrada
model = torch.jit.script(torch.nn.Conv2d(3, 16, 3, 1, 1)).eval()  # CPU
dummy = torch.randn(1, 3, 640, 640)

# Warmup
for _ in range(50):
    _ = model(dummy)

# Medición
times = []
for _ in range(200):
    t0 = time.time()
    _ = model(dummy)
    times.append((time.time() - t0) * 1000)  # ms

p95 = sorted(times)[int(0.95 * len(times)) - 1]
fps = 1000.0 / (sum(times) / len(times))
print(f"CPU  FPS={fps:.1f}   p95(ms)={p95:.2f}")
