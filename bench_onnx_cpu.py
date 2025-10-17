import onnxruntime as ort, numpy as np, time
sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
dummy = np.random.randn(1,3,640,640).astype("float32")
for _ in range(50): sess.run(None, {"images": dummy})
ts=[]
for _ in range(200):
    t0=time.time(); sess.run(None, {"images": dummy}); ts.append((time.time()-t0)*1000)
p95 = sorted(ts)[int(0.95*len(ts))-1]; fps = 1000.0/(sum(ts)/len(ts))
print("FPS=",fps," p95(ms)=",p95)
