from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

NOISE_LEVEL_MIN_RELATIVE_TO_PEAK = 1e-3
NOISE_LEVEL_MIN = 1e-5
MAX_SEGMENT_SECONDS = 12.0
MIN_PENDING_SECONDS = 2.0
STEP_SECONDS = 1.0
STITCH_LEFT_CONTEXT_SECONDS = 4.0
STITCH_FREEZE_GUARD_SECONDS = 2.0
STITCH_LEFT_RAMP_SECONDS = 2.0
STITCH_RIGHT_RAMP_SECONDS = 1.0
STITCH_DILATION_FRAMES = 1
WORD_BOUNDARY_BLANK_HOPS = 134
MIN_VIEW_WEIGHT = 1e-3
UNCOVERED_LOG_PROB = -30.0


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    """Read PCM WAV audio and return mono float32 samples in the range [-1, 1]."""
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        raw = wav.readframes(frame_count)

    if sample_width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return sample_rate, audio.astype(np.float32, copy=False)


def resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample with linear interpolation to keep the example dependency-light."""
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    if len(audio) == 0:
        return audio.astype(np.float32, copy=False)

    target_length = int(round(len(audio) * target_rate / source_rate))
    source_positions = np.arange(target_length, dtype=np.float64) * source_rate / target_rate
    left = np.floor(source_positions).astype(np.int64)
    right = np.minimum(left + 1, len(audio) - 1)
    fraction = (source_positions - left).astype(np.float32)
    resampled = audio[left] * (1.0 - fraction) + audio[right] * fraction
    return resampled.astype(np.float32, copy=False)


def frequency_bin_range(sample_rate: int, fft_length: int, min_hz: float, max_hz: float) -> tuple[int, int]:
    bin_hz = sample_rate / fft_length
    start_bin = int(math.ceil(min_hz / bin_hz))
    stop_bin = int(math.floor(max_hz / bin_hz)) + 1
    return start_bin, stop_bin


def audio_to_spectrogram(audio: np.ndarray, metadata: dict) -> np.ndarray:
    """Create a [1, 1, time, frequency] float32 spectrogram for the ONNX model."""
    fft_length = int(metadata["fft_length"])
    hop_length = int(metadata["hop_length"])
    sample_rate = int(metadata["sample_rate"])
    min_hz = float(metadata["spectrogram_min_freq_hz"])
    max_hz = float(metadata["spectrogram_max_freq_hz"])
    expected_bins = int(metadata["spectrogram_frequency_bins"])

    if len(audio) < fft_length:
        raise ValueError(f"Audio is too short for fft_length={fft_length}")

    start_bin, stop_bin = frequency_bin_range(sample_rate, fft_length, min_hz, max_hz)
    if stop_bin - start_bin != expected_bins:
        raise ValueError(f"Metadata expects {expected_bins} bins, but the computed range has {stop_bin - start_bin}")

    pad = fft_length // 2
    audio = np.pad(audio, (pad, pad), mode="reflect")
    window = np.hanning(fft_length + 1)[:-1].astype(np.float32)
    frames = 1 + (len(audio) - fft_length) // hop_length
    spectrogram = np.empty((frames, expected_bins), dtype=np.float32)

    for frame_index in range(frames):
        start = frame_index * hop_length
        frame = audio[start : start + fft_length] * window
        spectrum = np.fft.rfft(frame, n=fft_length)
        spectrogram[frame_index] = np.abs(spectrum[start_bin:stop_bin]).astype(np.float32)

    return normalize_spectrogram(spectrogram, metadata)[np.newaxis, np.newaxis, :, :]


def normalize_spectrogram(spectrogram: np.ndarray, metadata: dict) -> np.ndarray:
    normalization = metadata.get("normalization")
    if not (isinstance(normalization, str) and normalization.startswith("noise_q") and normalization.endswith("_log1p")):
        raise ValueError(f"Unsupported normalization: {normalization}")

    quantile = float(metadata.get("noise_level_quantile", 0.9))
    flat = np.sort(spectrogram.reshape(-1))
    if len(flat) == 0:
        return spectrogram.astype(np.float32, copy=False)
    rayleigh_factor = math.sqrt(-2.0 * math.log(1.0 - quantile))
    noise_level = max(
        float(flat[math.floor(quantile * (len(flat) - 1))]) / rayleigh_factor,
        float(flat[-1]) * NOISE_LEVEL_MIN_RELATIVE_TO_PEAK,
        NOISE_LEVEL_MIN,
    )
    return np.log1p(spectrogram / np.float32(noise_level)).astype(np.float32, copy=False)



@dataclass
class Span:
    char: str
    start_frame: int
    end_frame: int


@dataclass
class LaneText:
    text: str
    prepend_space: bool


def smoothstep(value: np.ndarray) -> np.ndarray:
    t = np.clip(value, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def log_softmax(values: np.ndarray) -> np.ndarray:
    peak = values.max(axis=-1, keepdims=True)
    return values - (peak + np.log(np.exp(values - peak).sum(axis=-1, keepdims=True)))


def to_dilated_log_probs(logits: np.ndarray, blank_index: int) -> np.ndarray:
    log_probs = log_softmax(logits.astype(np.float32))
    radius = STITCH_DILATION_FRAMES
    padded = np.pad(log_probs, ((radius, radius), (0, 0)), constant_values=-np.inf)
    dilated = np.max([padded[offset : offset + len(log_probs)] for offset in range(2 * radius + 1)], axis=0)
    dilated[:, blank_index] = log_probs[:, blank_index]
    return log_softmax(dilated)


def decode_ctc_greedy(
    scores: np.ndarray,
    chars: list[str],
    blank_index: int,
    space_index: int,
    word_boundary_frames: int,
    previous_symbol: str,
    blank_frames: int,
) -> tuple[list[Span], np.ndarray]:
    argmax = scores.argmax(axis=-1)
    indices = argmax.copy()
    frame_count = len(indices)

    if space_index >= 0:
        previous = -2 if previous_symbol == "char" else space_index if previous_symbol == "space" else -1
        frame = 0
        while frame < frame_count:
            if indices[frame] != blank_index:
                previous = int(indices[frame])
                frame += 1
                continue
            run_start = frame
            while frame < frame_count and indices[frame] == blank_index:
                frame += 1
            following = int(indices[frame]) if frame < frame_count else -1
            run_length = frame - run_start + (blank_frames if run_start == 0 else 0)
            if (
                run_length >= word_boundary_frames
                and previous not in (-1, space_index)
                and following >= 0
                and following != space_index
            ):
                indices[run_start:frame] = space_index

    spans: list[Span] = []
    previous_index = -1
    active: Span | None = None
    for frame, index in enumerate(indices):
        index = int(index)
        if index == blank_index:
            previous_index = -1
            active = None
            continue
        if index == previous_index:
            if active is not None:
                active.end_frame = frame
            continue
        previous_index = index
        if index >= len(chars):
            active = None
            continue
        active = Span(chars[index], frame, frame)
        spans.append(active)

    return spans, argmax


def to_lane_text(spans: list[Span], prepend_space: bool) -> LaneText:
    output: list[str] = []
    has_leading_space = prepend_space
    pending_space = False
    for span in spans:
        if span.char.isspace():
            if not output:
                has_leading_space = True
            else:
                pending_space = True
            continue
        if pending_space:
            output.append(" ")
            pending_space = False
        output.append(span.char)
    return LaneText("".join(output), has_leading_space)


class PosteriorStitcher:
    def __init__(self, metadata: dict) -> None:
        self.chars = list(metadata["chars"])
        self.blank_index = int(metadata["blank_index"])
        self.space_index = self.chars.index(" ") if " " in self.chars else -1
        self.class_count = int(metadata["num_classes"])
        output_time_stride = int(metadata.get("output_time_stride", 1))
        self.hop_length = int(metadata["hop_length"]) * output_time_stride
        sample_rate = int(metadata["sample_rate"])
        self.frames_per_second = sample_rate / self.hop_length
        self.max_segment_samples = int(MAX_SEGMENT_SECONDS * sample_rate)
        self.left_context_frames = round(STITCH_LEFT_CONTEXT_SECONDS * sample_rate / self.hop_length)
        self.freeze_guard_frames = round(STITCH_FREEZE_GUARD_SECONDS * sample_rate / self.hop_length)
        self.word_boundary_frames = math.ceil(WORD_BOUNDARY_BLANK_HOPS / output_time_stride)

        self.row_base = 0
        self.sums = np.zeros((0, self.class_count), dtype=np.float32)
        self.weights = np.zeros(0, dtype=np.float32)
        self.covered_end = 0
        self.frozen_end = 0
        self.committed_end = 0
        self.previous_symbol = "none"
        self.blank_frames = 0
        self.has_committed_text = False

    @property
    def context_start_frame(self) -> int:
        return max(0, self.frozen_end - self.left_context_frames)

    def window_start_frame(self, end_sample: int) -> int:
        return max(self.context_start_frame, math.ceil((end_sample - self.max_segment_samples) / self.hop_length))

    def add_window(self, start_frame: int, logits: np.ndarray, is_final: bool) -> None:
        frame_count = len(logits)
        if frame_count <= 0:
            if is_final:
                self.frozen_end = self.covered_end
            return

        end_frame = start_frame + frame_count
        self._ensure_rows(end_frame)
        log_probs = to_dilated_log_probs(logits, self.blank_index)
        frames = np.arange(max(0, self.frozen_end - start_frame), frame_count)
        left_seconds = np.inf if start_frame == 0 else frames / self.frames_per_second
        right_seconds = np.inf if is_final else (frame_count - 1 - frames) / self.frames_per_second
        weights = np.maximum(
            MIN_VIEW_WEIGHT,
            smoothstep(left_seconds / STITCH_LEFT_RAMP_SECONDS) * smoothstep(right_seconds / STITCH_RIGHT_RAMP_SECONDS),
        ).astype(np.float32)
        weights = np.broadcast_to(weights, frames.shape)
        rows = start_frame + frames - self.row_base
        self.sums[rows] += weights[:, None] * log_probs[frames]
        self.weights[rows] += weights

        self.covered_end = max(self.covered_end, end_frame)
        if is_final:
            self.frozen_end = self.covered_end
        else:
            self.frozen_end = max(self.frozen_end, end_frame - self.freeze_guard_frames)

    def take(self, final: bool = False) -> tuple[LaneText | None, LaneText]:
        start_frame = self.committed_end
        if self.covered_end - start_frame <= 0:
            return None, LaneText("", False)

        starts_word = not self.has_committed_text or self.previous_symbol == "space"
        spans, argmax = decode_ctc_greedy(
            self._read_scores(start_frame, self.covered_end),
            self.chars,
            self.blank_index,
            self.space_index,
            self.word_boundary_frames,
            self.previous_symbol,
            self.blank_frames,
        )
        for span in spans:
            span.start_frame += start_frame
            span.end_frame += start_frame

        commit_limit = math.inf if final else self.frozen_end
        committed_count = 0
        for index, span in enumerate(spans):
            if span.end_frame + 1 >= commit_limit:
                break
            if not span.char.isspace():
                committed_count = index + 1

        committed: LaneText | None = None
        if committed_count > 0:
            committed = to_lane_text(spans[:committed_count], starts_word)
            self.has_committed_text = True
            self.committed_end = spans[committed_count - 1].end_frame + 1
            self.previous_symbol = "char"
            self.blank_frames = 0

        remaining = spans[committed_count:]
        next_character = next((span for span in remaining if not span.char.isspace()), None)
        release_end = (
            min(
                self.covered_end if final else self.frozen_end,
                next_character.start_frame if next_character else math.inf,
            )
            - 1
        )
        previous_symbol, blank_frames = self.previous_symbol, self.blank_frames
        frame = self.committed_end
        while frame < release_end:
            if argmax[frame - start_frame] == self.space_index:
                previous_symbol = "space"
                blank_frames = 0
            else:
                blank_frames += 1
            frame += 1
        if release_end > self.committed_end:
            self.previous_symbol, self.blank_frames = previous_symbol, blank_frames
            self.committed_end = int(release_end)
        self._drop_rows_before(self.committed_end)

        pending_spans = [span for span in remaining if span.end_frame >= self.committed_end]
        pending = to_lane_text(pending_spans, not self.has_committed_text or self.previous_symbol == "space")
        return committed, pending

    def _ensure_rows(self, end_frame: int) -> None:
        missing = end_frame - self.row_base - len(self.weights)
        if missing <= 0:
            return
        self.sums = np.concatenate([self.sums, np.zeros((missing, self.class_count), dtype=np.float32)])
        self.weights = np.concatenate([self.weights, np.zeros(missing, dtype=np.float32)])

    def _drop_rows_before(self, frame: int) -> None:
        drop_count = min(frame - self.row_base, len(self.weights))
        if drop_count <= 0:
            return
        self.sums = self.sums[drop_count:]
        self.weights = self.weights[drop_count:]
        self.row_base += drop_count

    def _read_scores(self, start_frame: int, end_frame: int) -> np.ndarray:
        rows = slice(start_frame - self.row_base, end_frame - self.row_base)
        weights = self.weights[rows]
        scores = np.full((end_frame - start_frame, self.class_count), UNCOVERED_LOG_PROB, dtype=np.float32)
        scores[:, self.blank_index] = 0.0
        covered = weights > 0
        scores[covered] = self.sums[rows][covered] / weights[covered, None]
        return scores


class Decoder:
    def __init__(self, model_path: Path, metadata: dict) -> None:
        self.metadata = metadata
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.stitcher = PosteriorStitcher(metadata)
        self.transcript = ""

    def run_window(self, audio: np.ndarray, start_frame: int, end_sample: int, is_final: bool) -> str:
        start_sample = start_frame * self.stitcher.hop_length
        spectrogram = audio_to_spectrogram(audio[start_sample:end_sample], self.metadata)
        outputs = self.session.run([self.metadata["onnx_output_name"]], {self.metadata["onnx_input_name"]: spectrogram})
        self.stitcher.add_window(start_frame, outputs[0][0], is_final)
        committed, _ = self.stitcher.take(is_final)
        if committed is None or not committed.text:
            return ""
        text = (" " if committed.prepend_space and self.transcript else "") + committed.text
        self.transcript += text
        return text

    def step(self, audio: np.ndarray, end_sample: int) -> str:
        return self.run_window(audio, self.stitcher.window_start_frame(end_sample), end_sample, False)

    def finalize(self, audio: np.ndarray) -> str:
        output = ""
        while True:
            start_frame = self.stitcher.context_start_frame
            end_sample = min(len(audio), start_frame * self.stitcher.hop_length + self.stitcher.max_segment_samples)
            is_final = end_sample >= len(audio)
            output += self.run_window(audio, start_frame, end_sample, is_final)
            if is_final:
                return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode Morse code audio with the DeepCW ONNX model.")
    parser.add_argument("--model", type=Path, default=Path("../../model.onnx"), help="Path to model.onnx.")
    parser.add_argument("--metadata", type=Path, default=Path("../../model.onnx.json"), help="Path to model metadata JSON.")
    parser.add_argument("--wav", type=Path, required=True, help="Path to a PCM WAV file of any length.")
    args = parser.parse_args()

    metadata = load_metadata(args.metadata)
    sample_rate = int(metadata["sample_rate"])
    source_rate, audio = read_wav_mono(args.wav)
    audio = resample_linear(audio, source_rate, sample_rate)

    decoder = Decoder(args.model, metadata)
    step_samples = int(STEP_SECONDS * sample_rate)
    end_sample = int(MIN_PENDING_SECONDS * sample_rate)
    while end_sample < len(audio):
        sys.stdout.write(decoder.step(audio, end_sample))
        sys.stdout.flush()
        end_sample += step_samples
    sys.stdout.write(decoder.finalize(audio) + "\n")


if __name__ == "__main__":
    main()
