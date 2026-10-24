# deepcw-engine

This repository contains the DeepCW model and a minimal example.

`model_narrow.onnx` is a faster, narrow-band model for multi-channel decoding. Use it with
`--model ../../model_narrow.onnx --metadata ../../model_narrow.onnx.json`.

## Preparing WAV Audio

The examples can read and resample ordinary PCM WAV files, but preparing the audio with
`ffmpeg` makes the input explicit and repeatable:

```bash
ffmpeg -i input.wav -ac 1 -ar 3200 -sample_fmt s16 target.wav
```

The sample rate shown above matches the current `model.onnx.json`. If the metadata changes,
use the `sample_rate` value from that file.

## Python Example

Directory: `examples/python`

Install dependencies:

```bash
cd examples/python
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the decoder:

```bash
python decode_morse.py \
  --model ../../model.onnx \
  --metadata ../../model.onnx.json \
  --wav test.wav
```

## Node.js Example

Directory: `examples/nodejs`

Install dependencies:

```bash
cd examples/nodejs
npm install
```

Run the decoder:

```bash
npm run decode -- \
  --model ../../model.onnx \
  --metadata ../../model.onnx.json \
  --wav test.wav
```

## License

AGPL-3.0-only. See [LICENSE](LICENSE).
