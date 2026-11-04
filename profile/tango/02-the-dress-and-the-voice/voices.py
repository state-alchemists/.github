#!/usr/bin/env python3
"""Make the emotion voices zrb speaks in: one short clip per emotion.

Pocket TTS clones a voice from a clip, so the clip *is* the voice, and the
clip's tone is the emotion the reply carries. Each one is rendered by
gpt-4o-mini-tts with an instruction, then resampled upward — which raises the
pitch and shortens the clip, the naive way, and the way these six files were
made in the first place.

    pip install openai                 # only for the render step
    export OPENAI_API_KEY=sk-...
    python voices.py                   # -> ./voices/{neutral,happy,...}.wav

    # no API key needed: just pitch a clip you already have
    python voices.py --shift in.wav out.wav --semitones 5

24 kHz, 16-bit, mono, a few seconds: what Pocket takes, and what the six files
in this folder measure.
"""

import argparse
import array
import os
import sys
import wave

MODEL = "gpt-4o-mini-tts"
VOICE = "nova"
SEMITONES = 5.0

# The words never change, so a difference between two of these files is a
# difference in delivery and nothing else. Long enough to clone a voice from
# (Pocket wants a few seconds), short enough to listen to while testing.
LINE = (
    "Hi, I'm Tara. I read through your notes before you got here, and I think "
    "I found the problem. Shall I show you, or do you want to guess first?"
)

# The instruction is the emotion. `instructions` is what the `tts-1` models do
# not take and `gpt-4o-mini-tts` does: a direction, not text to be read.
INSTRUCTIONS = {
    "neutral": "Even and clear, a colleague explaining something. No performance.",
    "happy": "Warm and pleased, a small smile in the voice. Bright, not loud.",
    "sad": "Quiet and slowed, the way someone talks when they are tired and disappointed.",
    "angry": "Clipped and low. The annoyance is held in, not raised.",
    "surprised": "Lifted and a little quicker, as though you did not expect to be asked.",
    "relaxed": "Soft and unhurried, comfortable in the room, faintly amused.",
}


def resample(samples: array.array, factor: float) -> array.array:
    """Read the samples *factor* apart, interpolating: pitch up, clip shorter.

    A phase vocoder would keep the length; this does not, and does not pretend
    to. On a six-second clip that nobody will time, it is the honest trade.
    """
    out_len = max(1, int(len(samples) / factor))
    out = array.array("h", bytes(2 * out_len))
    last = len(samples) - 1
    for i in range(out_len):
        pos = i * factor
        j = int(pos)
        frac = pos - j
        a = samples[min(j, last)]
        b = samples[min(j + 1, last)]
        out[i] = int(a + (b - a) * frac)
    return out


def read_wav(path: str) -> tuple[array.array, int]:
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            sys.exit(
                f"{path}: need 16-bit mono, got "
                f"{w.getnchannels()}ch/{8 * w.getsampwidth()}-bit"
            )
        return array.array("h", w.readframes(w.getnframes())), w.getframerate()


def write_wav(path: str, samples: array.array, rate: int) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


def pitch(samples: array.array, semitones: float) -> array.array:
    """*semitones* up, at the same sample rate: the file plays higher and faster."""
    return resample(samples, 2 ** (semitones / 12.0))


def render(emotion: str, path: str, semitones: float) -> None:
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("pip install openai (only to render; --shift works without it)")

    raw = path + ".raw.wav"
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    client = OpenAI()
    with client.audio.speech.with_streaming_response.create(
        model=MODEL,
        voice=VOICE,
        input=LINE,
        instructions=INSTRUCTIONS[emotion],
        response_format="wav",
    ) as response:
        response.stream_to_file(raw)
    samples, rate = read_wav(raw)
    os.remove(raw)
    if rate != 24000:
        print(f"note: {emotion} came back at {rate} Hz, not 24000", file=sys.stderr)
    write_wav(path, pitch(samples, semitones), rate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="voices", help="folder to write the clips into")
    parser.add_argument("--semitones", type=float, default=SEMITONES)
    parser.add_argument(
        "--shift",
        nargs=2,
        metavar=("IN", "OUT"),
        help="pitch one clip and stop; needs no API key",
    )
    parser.add_argument("emotions", nargs="*", help="only these (default: all six)")
    args = parser.parse_args(argv)

    if args.shift:
        samples, rate = read_wav(args.shift[0])
        shifted = pitch(samples, args.semitones)
        write_wav(args.shift[1], shifted, rate)
        print(
            f"{args.shift[1]}: {len(samples) / rate:.2f}s -> "
            f"{len(shifted) / rate:.2f}s at {rate} Hz, +{args.semitones:g} semitones"
        )
        return 0

    for emotion in args.emotions or list(INSTRUCTIONS):
        if emotion not in INSTRUCTIONS:
            sys.exit(f"no such emotion: {emotion} (have: {', '.join(INSTRUCTIONS)})")
        path = os.path.join(args.out, f"{emotion}.wav")
        render(emotion, path, args.semitones)
        samples, rate = read_wav(path)
        print(f"{path}: {len(samples) / rate:.2f}s at {rate} Hz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
