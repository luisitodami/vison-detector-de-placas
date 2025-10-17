# CPU-only micro-benchmark (dummy Conv2d)
import time, torch
print('device = cpu')
model = torch.jit.script(torch.nn.Conv2d(3,16,3,1,1)).eval()
dummy = torch.randn(1,3,640,640)
for _ in range(50): _ = model(dummy)
ts=[]
for _ in range(200):
    t0=time.time(); _ = model(dummy); ts.append((time.time()-t0)*1000)
p95 = sorted(ts)[int(0.95*len(ts))-1]; fps = 1000.0/(sum(ts)/len(ts))
print(f'CPU  FPS={fps:.1f}   p95(ms)={p95:.2f}')