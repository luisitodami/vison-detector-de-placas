# ONNXRuntime CPU benchmark
import argparse, onnxruntime as ort, numpy as np, time, os
p=argparse.ArgumentParser()
p.add_argument('--onnx', required=True)
p.add_argument('--imgsz', type=int, default=640)
a=p.parse_args()
assert os.path.exists(a.onnx), f'ONNX not found: {a.onnx}'
sess = ort.InferenceSession(a.onnx, providers=['CPUExecutionProvider'])
name = sess.get_inputs()[0].name
dummy = np.random.randn(1,3,a.imgsz,a.imgsz).astype(np.float32)
for _ in range(50): _ = sess.run(None, {name: dummy})
ts=[]
for _ in range(200):
    t0=time.time(); _ = sess.run(None, {name: dummy}); ts.append((time.time()-t0)*1000)
p95 = sorted(ts)[int(0.95*len(ts))-1]; fps = 1000.0/(sum(ts)/len(ts))
print(f'ONNXRuntime CPU  FPS={fps:.1f}   p95(ms)={p95:.2f}')