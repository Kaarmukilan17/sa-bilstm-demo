# Performance Benchmarking of SA-BiLSTM Based Fake News Detection Using OpenVINO Framework

**Live demo (browser):** a static web page that runs the trained SA-BiLSTM fake news classifier directly in the visitor's browser and reports the latency and throughput of a batch of 32 articles.

School of Electronics Engineering (SENSE), Vellore Institute of Technology
B.Tech. Electronics & Communication Engineering

| Team member | Reg. no. |
|---|---|
| Kaarmukilan A G | 23BEC0137 |
| Rohith S | 23BEC0350 |
| Sanjay Elavarasan Karthikeyan | 23BEC0503 |

Guide: Dr. Sivasankaran K, Professor, Dept. of Micro & Nano Electronics, SENSE, VIT

---

## What this site does

- **Classify an article.** Paste a news article, or press *Random fake article*, *Random genuine article* or *Random (either)* to load a real article from the ISOT test split. The model returns the probability that the article is fake (0.50 or more is called fake). For random articles the page also shows the dataset's true label and whether the model was right.
- **Measure speed.** *Run batch of 32* sends 32 random test articles through the model in one call and reports mean, median and 95th-percentile latency (ms per batch), throughput (articles per second) and accuracy on those 32. The 32 articles and the model's answers are listed in a collapsible table.
- **Compare with our CPU benchmark.** A bar chart shows the batch-32 latency and throughput we measured for PyTorch, ONNX Runtime and OpenVINO (FP32, FP16). After a run, the visitor's own result is added as a striped bar.

### What "batch of 32" means
Instead of classifying articles one at a time, the model takes 32 articles together and processes them in a single call. We use 32 because it is the batch size of our CPU benchmark.

- **Latency** = time for that one call, in milliseconds, for the whole batch (not per article).
- **Throughput** = articles processed per second = 32 ÷ latency.

---

## The model

| Item | Value |
|---|---|
| Architecture | Embedding (10,000 × 100) → BiLSTM (100 per direction) → scaled self-attention → masked mean pooling → Dropout 0.3 → Linear (200 → 1) |
| Attention | Scaled dot-product self-attention over the BiLSTM outputs, one trainable scalar, padding masked |
| Parameters | 1,161,802 |
| ONNX file | 4.44 MB, opset 17, input `input_ids` (int64, `[batch, 300]`), output `logits` |
| Framework | PyTorch (CPU), exported to ONNX |
| Dataset | ISOT, 39,103 articles after removing duplicate texts (21,196 genuine, 17,907 fake) |
| Split | 70 / 15 / 15 stratified: 27,372 train, 5,865 validation, 5,866 test |
| Training | Adam, lr 1e-4, BCE-with-logits, batch 32, early stopping (patience 5) stopped at epoch 14, best validation loss 0.02488 at epoch 9 |

### Test results (5,866 held-out articles)

| Metric | Value |
|---|---|
| Accuracy | 99.25% |
| Precision | 99.44% |
| Sensitivity (recall) | 98.92% |
| Specificity | 99.53% |
| F1-score | 99.18% |

Confusion matrix: TN 3165, FP 15, FN 29, TP 2657.

### CPU benchmark (batch of 32)
Measured in Python on an Intel Core i5-1135G7 (11th Gen, 8 threads), 8 GB RAM, Windows 11, OpenVINO 2024.6. Mean latency per batch, same pre-tokenised inputs for every runtime.

| Runtime | Latency (ms / batch of 32) | Throughput (articles/s) |
|---|---|---|
| PyTorch (CPU) | 160.3 | 200 |
| ONNX Runtime (CPU) | 116.7 | 274 |
| OpenVINO FP32 (CPU) | 111.8 | 286 |
| OpenVINO FP16 (CPU) | 102.6 | 312 |

OpenVINO FP16 is the fastest at batch 32, about 1.56× faster than PyTorch (FP32: 1.43×, ONNX Runtime: 1.37×). At batch sizes 1 and 8, ONNX Runtime was the fastest.

---

## How the browser demo works

- The page loads `sa_bilstm.onnx` and runs it with **ONNX Runtime Web** (WebAssembly, v1.31.1) entirely on the visitor's device. No article text is sent anywhere.
- The OpenVINO `.xml` / `.bin` files are **not** used here: OpenVINO cannot run in a browser. The chart rows for OpenVINO and PyTorch are our measured results, not live runs.
- **Tokenizer (must match training):** lowercase, split into words with `\b\w+\b`, look each word up in `vocabulary.json` (unknown → 1), keep the first 300 tokens, pad with 0 at the end.
- **Timing:** tokenising is done before timing starts. Each batch run uses 3 warm-up runs and 20 timed runs of the same batch, so the numbers cover model inference only.
- **Threads:** when the host sends the cross-origin-isolation headers (`vercel.json` does this), WebAssembly uses up to 4 threads. Without them (for example on GitHub Pages) it runs single-threaded and the page says so.

---

## Repository layout

```
index.html              page structure
style.css               styling
app.js                  tokenizer, ONNX Runtime Web inference, timing, chart
vercel.json             headers that enable multi-threaded WebAssembly on Vercel
.nojekyll               lets GitHub Pages serve the files untouched
assets/
  sa_bilstm.onnx        exported model
  vocabulary.json       token → id map (10,000 entries)
  demo_samples.json     200 ISOT test articles (100 genuine, 100 fake)
  vit-logo.png          header logo
  ort/                  ONNX Runtime Web (JS loader + .wasm), served locally, no CDN
```

---

## Run locally

Opening `index.html` by double-click does **not** work: the browser blocks loading the model from a `file://` page. Serve the folder instead:

```
python -m http.server 8000
```

Then open <http://localhost:8000>. (A local server has no cross-origin-isolation headers, so it runs single-threaded.)

## Deploy

**Vercel (recommended)**
1. Push this folder to a GitHub repository (`index.html` at the repo root).
2. In Vercel: Add New → Project → import the repository.
3. Application Preset: **Other**. Root Directory: `./`. Leave build and output settings empty.
4. Deploy. After a batch run, the line under the stats should say how many WebAssembly threads were used.

**GitHub Pages**
Settings → Pages → Deploy from a branch → `main` / `(root)`. Works, but single-threaded, so latency is higher.

## Updating the demo articles

`assets/demo_samples.json` was built from `True.csv` and `Fake.csv` with the same preprocessing and split as the training notebook (seed 42), then:

```python
sample = pd.concat([
    test_df[test_df.label == 0].sample(100, random_state=SEED),
    test_df[test_df.label == 1].sample(100, random_state=SEED),
])
sample["text"] = sample["combined_text"].str.slice(0, 2500)
sample[["text", "label"]].to_json("demo_samples.json", orient="records")
```

If you retrain, replace `sa_bilstm.onnx` and `vocabulary.json` together, because the vocabulary and the model must come from the same run.

---

## Limitations

- **ISOT is easy.** Genuine ISOT articles carry strong source cues (for example agency bylines), so a 99% score on ISOT does not mean the model will catch fake news from other sources.
- **Short or non-news text is unreliable.** The model was trained only on full ISOT news articles.
- **Browser speed depends on the device.** Compare devices with each other, not with the OpenVINO benchmark rows: the browser uses a different runtime on different hardware.
- **Latency is per batch of 32.** Divide by 32 for an approximate per-article time.
- **FPGA stage is future work.** This project covers training, ONNX export and OpenVINO on CPU. Deployment on the Intel Agilex 7 FPGA board (via Intel FPGA AI Suite) is planned.

## Reference

W. Jian et al., "SA-Bi-LSTM: Self attention with bi-directional LSTM-based intelligent model for accurate fake news detection," *IEEE Access*, vol. 12, pp. 48436–48452, 2024.
