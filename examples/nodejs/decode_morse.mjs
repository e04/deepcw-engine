import fs from "node:fs";
import path from "node:path";
import * as ort from "onnxruntime-node";

const NOISE_LEVEL_MIN_RELATIVE_TO_PEAK = 1e-3;
const NOISE_LEVEL_MIN = 1e-5;

const MAX_SEGMENT_SECONDS = 12;
const MIN_PENDING_SECONDS = 2;
const STEP_SECONDS = 1;
const STITCH_LEFT_CONTEXT_SECONDS = 4;
const STITCH_FREEZE_GUARD_SECONDS = 2;
const STITCH_LEFT_RAMP_SECONDS = 2;
const STITCH_RIGHT_RAMP_SECONDS = 1;
const STITCH_DILATION_FRAMES = 1;
const WORD_BOUNDARY_BLANK_HOPS = 134;
const MIN_VIEW_WEIGHT = 1e-3;
const UNCOVERED_LOG_PROB = -30;

function parseArgs() {
  const args = {
    model: "../../model.onnx",
    metadata: "../../model.onnx.json",
    wav: null,
  };

  for (let i = 2; i < process.argv.length; i += 1) {
    const key = process.argv[i];
    const value = process.argv[i + 1];
    if (key === "--model" && value) args.model = value;
    if (key === "--metadata" && value) args.metadata = value;
    if (key === "--wav" && value) args.wav = value;
    if (key.startsWith("--")) i += 1;
  }

  if (!args.wav) {
    throw new Error("Missing required argument: --wav <path-to-pcm-wav>");
  }

  return args;
}

function readString(buffer, offset, length) {
  return buffer.toString("ascii", offset, offset + length);
}

function readWavMono(filePath) {
  const buffer = fs.readFileSync(filePath);
  if (readString(buffer, 0, 4) !== "RIFF" || readString(buffer, 8, 4) !== "WAVE") {
    throw new Error("Expected a RIFF/WAVE file.");
  }

  let offset = 12;
  let format = null;
  let dataOffset = -1;
  let dataSize = 0;

  while (offset + 8 <= buffer.length) {
    const chunkId = readString(buffer, offset, 4);
    const chunkSize = buffer.readUInt32LE(offset + 4);
    const chunkData = offset + 8;

    if (chunkId === "fmt ") {
      format = {
        audioFormat: buffer.readUInt16LE(chunkData),
        channels: buffer.readUInt16LE(chunkData + 2),
        sampleRate: buffer.readUInt32LE(chunkData + 4),
        bitsPerSample: buffer.readUInt16LE(chunkData + 14),
      };
    } else if (chunkId === "data") {
      dataOffset = chunkData;
      dataSize = chunkSize;
    }

    offset = chunkData + chunkSize + (chunkSize % 2);
  }

  if (!format || dataOffset < 0) {
    throw new Error("The WAV file is missing a fmt or data chunk.");
  }
  if (format.audioFormat !== 1) {
    throw new Error(`Only PCM WAV is supported by this compact example. Got format ${format.audioFormat}.`);
  }
  if (![8, 16, 32].includes(format.bitsPerSample)) {
    throw new Error(`Unsupported PCM bit depth: ${format.bitsPerSample}`);
  }

  const bytesPerSample = format.bitsPerSample / 8;
  const frameCount = Math.floor(dataSize / (bytesPerSample * format.channels));
  const audio = new Float32Array(frameCount);

  for (let frame = 0; frame < frameCount; frame += 1) {
    let sum = 0;
    for (let channel = 0; channel < format.channels; channel += 1) {
      const sampleOffset = dataOffset + (frame * format.channels + channel) * bytesPerSample;
      if (format.bitsPerSample === 8) sum += (buffer.readUInt8(sampleOffset) - 128) / 128;
      if (format.bitsPerSample === 16) sum += buffer.readInt16LE(sampleOffset) / 32768;
      if (format.bitsPerSample === 32) sum += buffer.readInt32LE(sampleOffset) / 2147483648;
    }
    audio[frame] = sum / format.channels;
  }

  return { sampleRate: format.sampleRate, audio };
}

function resampleLinear(audio, sourceRate, targetRate) {
  if (sourceRate === targetRate) return audio;
  const targetLength = Math.round((audio.length * targetRate) / sourceRate);
  const output = new Float32Array(targetLength);

  for (let i = 0; i < targetLength; i += 1) {
    const sourcePosition = (i * sourceRate) / targetRate;
    const left = Math.floor(sourcePosition);
    const right = Math.min(left + 1, audio.length - 1);
    const fraction = sourcePosition - left;
    output[i] = audio[left] * (1 - fraction) + audio[right] * fraction;
  }

  return output;
}

function frequencyBinRange(sampleRate, fftLength, minHz, maxHz) {
  const binHz = sampleRate / fftLength;
  return {
    startBin: Math.ceil(minHz / binHz),
    stopBin: Math.floor(maxHz / binHz) + 1,
  };
}

function selectedDftMagnitudes(frame, startBin, stopBin) {
  const length = frame.length;
  const output = new Float32Array(stopBin - startBin);

  for (let bin = startBin; bin < stopBin; bin += 1) {
    let real = 0;
    let imaginary = 0;
    for (let n = 0; n < length; n += 1) {
      const angle = (-2 * Math.PI * bin * n) / length;
      real += frame[n] * Math.cos(angle);
      imaginary += frame[n] * Math.sin(angle);
    }
    output[bin - startBin] = Math.hypot(real, imaginary);
  }

  return output;
}

function audioToSpectrogram(audio, metadata) {
  const fftLength = metadata.fft_length;
  const hopLength = metadata.hop_length;
  const expectedBins = metadata.spectrogram_frequency_bins;
  if (audio.length < fftLength) throw new Error(`Audio is too short for fft_length=${fftLength}.`);

  const { startBin, stopBin } = frequencyBinRange(
    metadata.sample_rate,
    fftLength,
    metadata.spectrogram_min_freq_hz,
    metadata.spectrogram_max_freq_hz,
  );
  if (stopBin - startBin !== expectedBins) {
    throw new Error(`Metadata expects ${expectedBins} bins, but computed ${stopBin - startBin}.`);
  }

  const pad = Math.floor(fftLength / 2);
  const padded = new Float32Array(audio.length + pad * 2);
  for (let i = 0; i < pad; i += 1) {
    padded[i] = audio[pad - i];
    padded[pad + audio.length + i] = audio[audio.length - 2 - i];
  }
  padded.set(audio, pad);

  const frames = 1 + Math.floor((padded.length - fftLength) / hopLength);
  const tensorData = new Float32Array(frames * expectedBins);
  const window = new Float32Array(fftLength);
  for (let i = 0; i < fftLength; i += 1) {
    window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / fftLength);
  }

  const frame = new Float32Array(fftLength);
  for (let frameIndex = 0; frameIndex < frames; frameIndex += 1) {
    const start = frameIndex * hopLength;
    for (let i = 0; i < fftLength; i += 1) {
      frame[i] = padded[start + i] * window[i];
    }
    const magnitudes = selectedDftMagnitudes(frame, startBin, stopBin);
    for (let bin = 0; bin < expectedBins; bin += 1) {
      tensorData[frameIndex * expectedBins + bin] = magnitudes[bin];
    }
  }

  normalizeSpectrogram(tensorData, metadata);
  return new ort.Tensor("float32", tensorData, [1, 1, frames, expectedBins]);
}

function normalizeSpectrogram(spectrogram, metadata) {
  const normalization = metadata.normalization;
  if (!(typeof normalization === "string" && normalization.startsWith("noise_q") && normalization.endsWith("_log1p"))) {
    throw new Error(`Unsupported normalization: ${normalization}`);
  }
  if (spectrogram.length === 0) return spectrogram;

  const quantile = metadata.noise_level_quantile ?? 0.9;
  const sorted = Float32Array.from(spectrogram).sort();
  const rayleighFactor = Math.sqrt(-2 * Math.log(1 - quantile));
  const noiseLevel = Math.max(
    sorted[Math.floor(quantile * (sorted.length - 1))] / rayleighFactor,
    sorted[sorted.length - 1] * NOISE_LEVEL_MIN_RELATIVE_TO_PEAK,
    NOISE_LEVEL_MIN,
  );
  for (let i = 0; i < spectrogram.length; i += 1) {
    spectrogram[i] = Math.log1p(spectrogram[i] / noiseLevel);
  }
  return spectrogram;
}

function smoothstep(value) {
  const t = Math.max(0, Math.min(1, value));
  return t * t * (3 - 2 * t);
}

function getViewWeight(leftSeconds, rightSeconds) {
  return Math.max(
    MIN_VIEW_WEIGHT,
    smoothstep(leftSeconds / STITCH_LEFT_RAMP_SECONDS) * smoothstep(rightSeconds / STITCH_RIGHT_RAMP_SECONDS),
  );
}

function logSoftmaxInPlace(values, offset, classCount) {
  let max = -Infinity;
  for (let index = 0; index < classCount; index += 1) {
    max = Math.max(max, values[offset + index]);
  }
  let sum = 0;
  for (let index = 0; index < classCount; index += 1) {
    sum += Math.exp(values[offset + index] - max);
  }
  const logSum = max + Math.log(sum);
  for (let index = 0; index < classCount; index += 1) {
    values[offset + index] -= logSum;
  }
}

function toDilatedLogProbs(logits, frameCount, classCount, blankIndex) {
  const logProbs = Float32Array.from(logits.subarray(0, frameCount * classCount));
  for (let frame = 0; frame < frameCount; frame += 1) {
    logSoftmaxInPlace(logProbs, frame * classCount, classCount);
  }

  const dilated = new Float32Array(logProbs.length);
  for (let frame = 0; frame < frameCount; frame += 1) {
    const offset = frame * classCount;
    const first = Math.max(0, frame - STITCH_DILATION_FRAMES);
    const last = Math.min(frameCount - 1, frame + STITCH_DILATION_FRAMES);
    for (let index = 0; index < classCount; index += 1) {
      let value = logProbs[offset + index];
      if (index !== blankIndex) {
        for (let other = first; other <= last; other += 1) {
          value = Math.max(value, logProbs[other * classCount + index]);
        }
      }
      dilated[offset + index] = value;
    }
    logSoftmaxInPlace(dilated, offset, classCount);
  }
  return dilated;
}

function decodeCtcGreedy(scores, frameCount, classCount, vocabulary, context) {
  const { blankIndex, spaceIndex, chars, wordBoundaryFrames } = vocabulary;
  const argmax = new Int32Array(frameCount);
  for (let frame = 0; frame < frameCount; frame += 1) {
    const offset = frame * classCount;
    let best = 0;
    for (let index = 1; index < classCount; index += 1) {
      if (scores[offset + index] > scores[offset + best]) best = index;
    }
    argmax[frame] = best;
  }

  const indices = argmax.slice();
  if (spaceIndex >= 0) {
    let previous = context.previousSymbol === "char" ? -2 : context.previousSymbol === "space" ? spaceIndex : -1;
    let frame = 0;
    while (frame < frameCount) {
      if (indices[frame] !== blankIndex) {
        previous = indices[frame];
        frame += 1;
        continue;
      }
      const runStart = frame;
      while (frame < frameCount && indices[frame] === blankIndex) {
        frame += 1;
      }
      const next = frame < frameCount ? indices[frame] : -1;
      const runLength = frame - runStart + (runStart === 0 ? context.blankFrames : 0);
      if (
        runLength >= wordBoundaryFrames &&
        previous !== -1 &&
        previous !== spaceIndex &&
        next >= 0 &&
        next !== spaceIndex
      ) {
        indices.fill(spaceIndex, runStart, frame);
      }
    }
  }

  const spans = [];
  let previousIndex = -1;
  let activeSpan = null;
  for (let frame = 0; frame < frameCount; frame += 1) {
    const index = indices[frame];
    if (index === blankIndex) {
      previousIndex = -1;
      activeSpan = null;
      continue;
    }
    if (index === previousIndex) {
      if (activeSpan) activeSpan.endFrame = frame;
      continue;
    }
    previousIndex = index;
    const char = chars[index];
    if (!char) {
      activeSpan = null;
      continue;
    }
    activeSpan = { char, startFrame: frame, endFrame: frame };
    spans.push(activeSpan);
  }

  return { spans, argmax };
}

function toLaneText(spans, prependSpace) {
  let text = "";
  let hasLeadingSpace = prependSpace;
  let pendingSpace = false;
  for (const span of spans) {
    if (/\s/.test(span.char)) {
      if (text.length === 0) {
        hasLeadingSpace = true;
      } else {
        pendingSpace = true;
      }
      continue;
    }
    if (pendingSpace) {
      text += " ";
      pendingSpace = false;
    }
    text += span.char;
  }
  return { text, prependSpace: hasLeadingSpace };
}

class PosteriorStitcher {
  constructor(metadata) {
    const outputTimeStride = metadata.output_time_stride ?? 1;
    this.hopLength = metadata.hop_length * outputTimeStride;
    this.framesPerSecond = metadata.sample_rate / this.hopLength;
    this.maxSegmentSamples = Math.round(MAX_SEGMENT_SECONDS * metadata.sample_rate);
    this.leftContextFrames = Math.round((STITCH_LEFT_CONTEXT_SECONDS * metadata.sample_rate) / this.hopLength);
    this.freezeGuardFrames = Math.round((STITCH_FREEZE_GUARD_SECONDS * metadata.sample_rate) / this.hopLength);
    this.vocabulary = {
      chars: metadata.chars,
      blankIndex: metadata.blank_index,
      spaceIndex: metadata.chars.indexOf(" "),
      wordBoundaryFrames: Math.ceil(WORD_BOUNDARY_BLANK_HOPS / outputTimeStride),
    };
    this.classCount = metadata.num_classes;
    this.rowBase = 0;
    this.sums = new Float32Array(0);
    this.weights = new Float32Array(0);
    this.coveredEnd = 0;
    this.frozenEnd = 0;
    this.committedEnd = 0;
    this.context = { previousSymbol: "none", blankFrames: 0 };
    this.hasCommittedText = false;
  }

  get contextStartFrame() {
    return Math.max(0, this.frozenEnd - this.leftContextFrames);
  }

  getWindowStartFrame(endSample) {
    return Math.max(this.contextStartFrame, Math.ceil((endSample - this.maxSegmentSamples) / this.hopLength));
  }

  addWindow(startFrame, logits, frameCount, isFinal) {
    if (frameCount <= 0) {
      if (isFinal) this.frozenEnd = this.coveredEnd;
      return;
    }

    const classCount = this.classCount;
    const endFrame = startFrame + frameCount;
    this.ensureRows(endFrame);
    const logProbs = toDilatedLogProbs(logits, frameCount, classCount, this.vocabulary.blankIndex);
    for (let frame = Math.max(0, this.frozenEnd - startFrame); frame < frameCount; frame += 1) {
      const weight = getViewWeight(
        startFrame === 0 ? Infinity : frame / this.framesPerSecond,
        isFinal ? Infinity : (frameCount - 1 - frame) / this.framesPerSecond,
      );
      const row = startFrame + frame - this.rowBase;
      for (let index = 0; index < classCount; index += 1) {
        this.sums[row * classCount + index] += weight * logProbs[frame * classCount + index];
      }
      this.weights[row] += weight;
    }

    this.coveredEnd = Math.max(this.coveredEnd, endFrame);
    this.frozenEnd = isFinal ? this.coveredEnd : Math.max(this.frozenEnd, endFrame - this.freezeGuardFrames);
  }

  take(final = false) {
    const startFrame = this.committedEnd;
    const frameCount = this.coveredEnd - startFrame;
    if (frameCount <= 0) {
      return { committed: null, pending: toLaneText([], false) };
    }

    const startsWord = !this.hasCommittedText || this.context.previousSymbol === "space";
    const { spans: relativeSpans, argmax } = decodeCtcGreedy(
      this.readScores(startFrame, this.coveredEnd),
      frameCount,
      this.classCount,
      this.vocabulary,
      this.context,
    );
    const spans = relativeSpans.map((span) => ({
      ...span,
      startFrame: span.startFrame + startFrame,
      endFrame: span.endFrame + startFrame,
    }));
    const commitLimit = final ? Infinity : this.frozenEnd;
    let committedCount = 0;
    for (let index = 0; index < spans.length; index += 1) {
      if (spans[index].endFrame + 1 >= commitLimit) break;
      if (!/\s/.test(spans[index].char)) committedCount = index + 1;
    }

    let committed = null;
    if (committedCount > 0) {
      const committedSpans = spans.slice(0, committedCount);
      committed = toLaneText(committedSpans, startsWord);
      this.hasCommittedText = true;
      this.committedEnd = committedSpans[committedCount - 1].endFrame + 1;
      this.context = { previousSymbol: "char", blankFrames: 0 };
    }

    const remaining = spans.slice(committedCount);
    const nextCharacter = remaining.find((span) => !/\s/.test(span.char));
    const releaseEnd =
      Math.min(final ? this.coveredEnd : this.frozenEnd, nextCharacter?.startFrame ?? Infinity) - 1;
    let { previousSymbol, blankFrames } = this.context;
    for (let frame = this.committedEnd; frame < releaseEnd; frame += 1) {
      if (argmax[frame - startFrame] === this.vocabulary.spaceIndex) {
        previousSymbol = "space";
        blankFrames = 0;
      } else {
        blankFrames += 1;
      }
    }
    if (releaseEnd > this.committedEnd) {
      this.context = { previousSymbol, blankFrames };
      this.committedEnd = releaseEnd;
    }
    this.dropRowsBefore(this.committedEnd);

    const pending = toLaneText(
      remaining.filter((span) => span.endFrame >= this.committedEnd),
      !this.hasCommittedText || this.context.previousSymbol === "space",
    );
    return { committed, pending };
  }

  ensureRows(endFrame) {
    const rowCount = endFrame - this.rowBase;
    if (rowCount <= this.weights.length) return;
    const capacity = Math.max(rowCount, this.weights.length * 2, 256);
    const sums = new Float32Array(capacity * this.classCount);
    sums.set(this.sums);
    const weights = new Float32Array(capacity);
    weights.set(this.weights);
    this.sums = sums;
    this.weights = weights;
  }

  dropRowsBefore(frame) {
    const dropCount = Math.min(frame - this.rowBase, this.weights.length);
    if (dropCount <= 0) return;
    this.sums.copyWithin(0, dropCount * this.classCount);
    this.sums.fill(0, this.sums.length - dropCount * this.classCount);
    this.weights.copyWithin(0, dropCount);
    this.weights.fill(0, this.weights.length - dropCount);
    this.rowBase += dropCount;
  }

  readScores(startFrame, endFrame) {
    const classCount = this.classCount;
    const scores = new Float32Array((endFrame - startFrame) * classCount);
    for (let frame = startFrame; frame < endFrame; frame += 1) {
      const row = frame - this.rowBase;
      const targetOffset = (frame - startFrame) * classCount;
      const weight = this.weights[row];
      if (!(weight > 0)) {
        scores.fill(UNCOVERED_LOG_PROB, targetOffset, targetOffset + classCount);
        scores[targetOffset + this.vocabulary.blankIndex] = 0;
        continue;
      }
      for (let index = 0; index < classCount; index += 1) {
        scores[targetOffset + index] = this.sums[row * classCount + index] / weight;
      }
    }
    return scores;
  }
}

class Decoder {
  constructor(session, metadata) {
    this.session = session;
    this.metadata = metadata;
    this.stitcher = new PosteriorStitcher(metadata);
    this.transcript = "";
  }

  async runWindow(audio, startFrame, endSample, isFinal) {
    const startSample = startFrame * this.stitcher.hopLength;
    const spectrogram = audioToSpectrogram(audio.subarray(startSample, endSample), this.metadata);
    const outputs = await this.session.run({ [this.metadata.onnx_input_name]: spectrogram });
    const logProbs = outputs[this.metadata.onnx_output_name];
    this.stitcher.addWindow(startFrame, logProbs.data, logProbs.dims[1], isFinal);
    const { committed } = this.stitcher.take(isFinal);
    if (!committed?.text) return "";
    const text = (committed.prependSpace && this.transcript ? " " : "") + committed.text;
    this.transcript += text;
    return text;
  }

  step(audio, endSample) {
    return this.runWindow(audio, this.stitcher.getWindowStartFrame(endSample), endSample, false);
  }

  async finalize(audio) {
    let output = "";
    for (;;) {
      const startFrame = this.stitcher.contextStartFrame;
      const endSample = Math.min(
        audio.length,
        startFrame * this.stitcher.hopLength + this.stitcher.maxSegmentSamples,
      );
      const isFinal = endSample >= audio.length;
      output += await this.runWindow(audio, startFrame, endSample, isFinal);
      if (isFinal) return output;
    }
  }
}

async function main() {
  const args = parseArgs();
  const modelPath = path.resolve(process.cwd(), args.model);
  const metadataPath = path.resolve(process.cwd(), args.metadata);
  const wavPath = path.resolve(process.cwd(), args.wav);
  const metadata = JSON.parse(fs.readFileSync(metadataPath, "utf8"));
  const { sampleRate, audio } = readWavMono(wavPath);
  const resampled = resampleLinear(audio, sampleRate, metadata.sample_rate);

  const session = await ort.InferenceSession.create(modelPath);
  const decoder = new Decoder(session, metadata);
  const stepSamples = Math.floor(STEP_SECONDS * metadata.sample_rate);
  for (
    let endSample = Math.floor(MIN_PENDING_SECONDS * metadata.sample_rate);
    endSample < resampled.length;
    endSample += stepSamples
  ) {
    process.stdout.write(await decoder.step(resampled, endSample));
  }
  process.stdout.write(`${await decoder.finalize(resampled)}\n`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
