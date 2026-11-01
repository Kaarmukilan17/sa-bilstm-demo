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

## The work

We trained a Bi-LSTM model with self-attention (SA-BiLSTM) in PyTorch to tell genuine news from fake news on the ISOT dataset (39,103 articles after removing duplicates). The model has 1.16 million parameters and is 4.44 MB as an ONNX file.

The model was then exported to ONNX and run on a CPU with PyTorch, ONNX Runtime and OpenVINO (FP32 and FP16) to compare latency and throughput. This site runs the same ONNX model in the browser: classify an article (or pick a random fake or genuine one from the test set) and time a batch of 32 articles on your own device.

A batch of 32 means 32 articles processed together in one model call. Latency is the time for that call, and throughput is 32 divided by that latency.

## Results

**Accuracy** on 5,866 held-out test articles:

| Accuracy | Precision | Sensitivity | Specificity | F1-score |
|---|---|---|---|---|
| 99.25% | 99.44% | 98.92% | 99.53% | 99.18% |

Confusion matrix: TN 3165, FP 15, FN 29, TP 2657.

**Speed** on an Intel Core i5-1135G7 (8 threads), batch of 32:

| Runtime | Latency (ms per batch) | Throughput (articles/s) |
|---|---|---|
| PyTorch | 160.3 | 200 |
| ONNX Runtime | 116.7 | 274 |
| OpenVINO FP32 | 111.8 | 286 |
| OpenVINO FP16 | 102.6 | 312 |

OpenVINO FP16 is the fastest, about 1.56× faster than PyTorch.

## Notes

- Browser speed depends on your device, and the browser runs ONNX Runtime Web, not OpenVINO, so it is not directly comparable to the table above.
- ISOT articles have strong source cues, so 99% accuracy on ISOT does not guarantee the same on news from other sources.
- Future work: run the model on the Intel Agilex 7 FPGA board.

---

**About the website:** a small static page (HTML, CSS and JavaScript, no server) where you paste a news article or pick a random fake or genuine one from the ISOT test set. The model, running locally in your browser through ONNX Runtime Web, says whether it is fake or genuine, and a "Run batch of 32" button measures latency and throughput on your device and compares them with our CPU benchmark.
