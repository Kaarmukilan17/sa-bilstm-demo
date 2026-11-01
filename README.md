# Performance Benchmarking of SA-BiLSTM Based Fake News Detection Using OpenVINO Framework

School of Electronics Engineering (SENSE), VIT
Kaarmukilan A G (23BEC0137), Rohith S (23BEC0350), Sanjay Elavarasan Karthikeyan (23BEC0503)
Guide: Dr. Sivasankaran K

A static web demo of our trained SA-BiLSTM fake news classifier. The model runs in the visitor's browser (ONNX Runtime Web), so no server is needed and no text is sent anywhere.

## What it does
- **Classify an article:** paste text, or load a random fake or genuine article from the ISOT test set. Shows P(fake) and, for random articles, the true label.
- **Batch of 32:** runs 32 test articles through the model in one call and reports latency (ms per batch of 32) and throughput (articles per second = 32 ÷ latency).
- **Benchmark chart:** compares our CPU results for PyTorch, ONNX Runtime and OpenVINO with the visitor's own run.

## Model and results
Embedding → BiLSTM (100 per direction) → scaled self-attention → masked mean pooling → Linear. 1.16 M parameters, 4.44 MB (ONNX). Trained in PyTorch on ISOT (39,103 articles, 70/15/15 split, sequence length 300).

Test set (5,866 articles): accuracy 99.25%, precision 99.44%, recall 98.92%, F1 99.18%.

CPU benchmark, batch of 32 (Intel Core i5-1135G7, 8 threads, OpenVINO 2024.6):

| Runtime | Latency (ms) | Throughput (articles/s) |
|---|---|---|
| PyTorch | 160.3 | 200 |
| ONNX Runtime | 116.7 | 274 |
| OpenVINO FP32 | 111.8 | 286 |
| OpenVINO FP16 | 102.6 | 312 |

## Run locally
Double-clicking `index.html` does not work (the browser blocks loading the model from a file). Serve the folder:

```
python -m http.server 8000
```

then open http://localhost:8000.

## Deploy
- **Vercel (recommended):** import the repo, preset **Other**, root `./`, no build settings. Done.
- **GitHub Pages:** Settings → Pages → `main` / root.

## Why speed differs between hosts
Multi-threaded WebAssembly needs two headers, which `vercel.json` sends. Vercel therefore runs on several threads. `python -m http.server` and GitHub Pages do not send them, so the model runs on one thread and latency is about 2–3× higher. The page shows how many threads were used. Quote the Vercel numbers.

## Files
`index.html`, `style.css`, `app.js` (page and inference) · `vercel.json` (headers) · `assets/` (`sa_bilstm.onnx`, `vocabulary.json`, `demo_samples.json`, `ort/` runtime).

## Notes
- The OpenVINO `.xml`/`.bin` files are not used in the browser; the OpenVINO and PyTorch rows are our measured results.
- Browser speed depends on the device, so compare devices with each other, not with the OpenVINO rows.
- ISOT is an easy dataset and the model expects full news articles; very short or non-news text gives unreliable results.
- Future work: Intel Agilex 7 FPGA deployment.
