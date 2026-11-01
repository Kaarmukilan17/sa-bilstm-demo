# SA-BiLSTM fake news demo (ONNX Runtime Web)

Static site. No build step, no server. The model runs in the visitor's browser.

## Demo articles
`assets/demo_samples.json` is included: 200 articles (100 genuine, 100 fake) drawn with
`random_state=42` from the ISOT **test** split, rebuilt from True.csv / Fake.csv with the
same preprocessing and split as the training notebook (test split reproduced exactly:
5,866 articles, confusion matrix 3165 / 15 / 29 / 2657).

## Run locally
`python -m http.server 8000` in this folder, then open http://localhost:8000
(double-clicking index.html will not work: the page loads the model with fetch).

## Deploy
- **Vercel (recommended):** import the folder as a static project (framework: Other).
  `vercel.json` sends the two headers that let WebAssembly use several threads,
  so latency numbers are lower and more representative.
- **GitHub Pages:** works, but Pages cannot send those headers, so the model runs
  single-threaded and the page says so. Expect slower numbers.

## Files
- `assets/sa_bilstm.onnx`, `assets/vocabulary.json`: your exported model and vocabulary.
- `assets/ort/`: ONNX Runtime Web (WASM build), served locally, no CDN.
- `app.js`: tokenizer (same as training: lowercase, `\b\w+\b`, first 300 tokens,
  unknown = 1, padding 0 at the end), inference, timing, chart.
- The CPU table in `app.js` (`REFERENCE`) holds your notebook's batch-32 results.

## Checked
Probabilities from the browser match Python ONNX Runtime to the 3 decimals shown.
