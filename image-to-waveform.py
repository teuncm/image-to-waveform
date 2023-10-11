# Author: Teun Mathijssen (https://github.com/teuncm)
# Derived from https://www.youtube.com/watch?v=qeUAHHPt-LY

import argparse
import numpy as np
from PIL import Image
import soundfile as sf

INTERP_NN = "nn"
INTERP_LIN = "lin"
SAMPLERATE_STANDARD = 44100
DURATION_DEFAULT = 2
LIGHTNESS_THRESHOLD = 250


def main(args):
    if args.inAudio:
        sig, sampleRate = readSignal(args.inAudio)
        sig = normSignal(monoSignal(sig))
    else:
        sampleRate = args.sampleRate
        frequency = args.frequency if args.frequency else sampleRate / 2
        sig = generateSignal(frequency, args.duration, sampleRate)

    ts = getTimeframe(sig, sampleRate)

    imgArr = readImage(args.inImg)
    imgArr = normImage(thresholdImage(imgArr))

    botIdxs, topIdxs = getIdxs(imgArr)
    centers, amps = getParams(imgArr, botIdxs, topIdxs)
    centers, amps = normParams(imgArr, centers, amps)

    if args.interp == INTERP_NN:
        centers, amps = nnInterpParams(ts, centers, amps)
    elif args.interp == INTERP_LIN:
        centers, amps = linInterpParams(ts, centers, amps)

    modSig = modSignal(sig, centers, amps)

    writeSignal(modSig, sampleRate, args.outAudio)


def readSignal(fName):
    """Read an audio array from a file."""
    sig, sampleRate = sf.read(fName)

    return sig, sampleRate


def monoSignal(sig):
    """Turn an audio signal into mono."""
    sig = np.expand_dims(sig, -1)
    mono = np.squeeze(np.mean(sig, axis=1))

    return mono


def clipSignal(sig):
    """Hard clip an audio signal."""
    return np.clip(sig, -1, 1)


def normSignal(x):
    """Normalize an audio signal."""
    norm = np.max(np.abs(y))
    y = x / norm

    return y


def generateSignal(freq, tMax, sampleRate):
    """Generate a simple sine wave over the given timeframe."""
    # tMax may not coincide with a sample.
    # Correct for this here
    numSamples = int(np.floor(tMax * sampleRate) + 1)
    tMaxCorrected = (numSamples - 1) / sampleRate

    ts = np.linspace(0, tMaxCorrected, numSamples)
    sig = np.cos(freq * 2 * np.pi * ts)

    return sig


def writeSignal(sig, sampleRate, fName):
    """Write an audio signal to a file."""
    sf.write(fName, sig, sampleRate)


def getTimeframe(sig, sampleRate):
    """Get timeframe for an audio signal."""
    tMax = (sig.shape[0] - 1) / sampleRate
    nSamples = sig.shape[0]

    ts = np.linspace(0, tMax, nSamples)

    return ts


def readImage(fName):
    """Read an image array from a file."""
    # Read and convert to grayscale
    img = Image.open(fName).convert(mode="L")
    arr = np.array(img)

    return arr


def thresholdImage(arr):
    """Use a lightness threshold to convert image to binary array."""
    mask = arr < LIGHTNESS_THRESHOLD
    arr[mask] = 1
    arr[~mask] = 0

    return arr


def normImage(arr):
    """Normalize an image to make full use of the amplitude."""
    # Condense image to one column
    mask = np.max(arr, axis=1)

    # Get mask bounds
    botIdx, topIdx = getIdxs(mask)

    # Crop array to bounds
    arr = arr[botIdx : topIdx + 1, :]

    return arr


def modSignal(sig, centers, amps):
    """Modify a signal using the given centers and amplitudes."""
    mod = centers + amps * sig

    return mod


def getIdxs(arr):
    """Find the bottom and top indices of the mask per column."""
    maxIdx = arr.shape[0] - 1

    # Find first index from the bottom
    botIdxs = np.argmax(arr, axis=0)
    # Find first index from the top
    flippedArr = np.flip(arr, axis=0)
    topIdxs = maxIdx - np.argmax(flippedArr, axis=0)

    return botIdxs, topIdxs


def getParams(arr, botIdxs, topIdxs):
    """Derive oscillation parameters from the given indices.
    Parameters are calculated in array space."""
    # Amplitude (in indices) of oscillation
    amps = (topIdxs - botIdxs) / 2

    mask = np.max(arr, axis=0) != 0
    # In columns without mask, nullify amplitude
    amps[~mask] = 0
    # In columns with mask, add amplitude
    amps[mask] += 0.5

    # Calculate center of oscillation
    centers = (botIdxs + topIdxs) / 2

    return centers, amps


def normParams(arr, centers, amps):
    """Normalize oscillation parameters."""
    maxIdx = arr.shape[0] - 1
    norm = arr.shape[0] / 2

    # Shift centers to middle of arr
    centers = centers - maxIdx / 2

    # Images are always loaded upside-down
    centers = -centers

    # Normalize
    amps = amps / norm
    centers = centers / norm

    return centers, amps


def linInterpParams(ts, centers, amps):
    """Interpolate oscillation parameters over a given timeframe.
    Use linear interpolation."""
    xMax = ts[-1]
    numSamples = len(centers)

    # Stretch our parameter grid such that its length
    # matches the time grid.
    xs = np.linspace(0, xMax, numSamples)

    # Interpolate our parameters using the resolution
    # of ts.
    centers = np.interp(ts, xs, centers)
    amps = np.interp(ts, xs, amps)

    return centers, amps


def nnInterpParams(ts, centers, amps):
    """Interpolate oscillation parameters over a given timeframe.
    Use nearest neighbor interpolation."""
    xMax = ts[-1]
    numSamples = len(centers)
    dx = xMax / (numSamples - 1)

    # Generate lookup table for ts -> xs
    lut = np.round(ts / dx).astype(int)

    # Interpolate our parameters using the resolution
    # of ts.
    centers = centers[lut]
    amps = amps[lut]

    return centers, amps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("inImg", help="Input image filename", type=str)

    parser.add_argument("outAudio", help="Output audio filename", type=str)

    parser.add_argument(
        "-i",
        "--inAudio",
        help="Input audio filename",
        type=str,
    )

    parser.add_argument(
        "-d",
        "--duration",
        help="Length of generated audio in seconds",
        type=float,
        default=DURATION_DEFAULT,
    )

    parser.add_argument("-f", "--frequency", help="Generator frequency", type=float)

    parser.add_argument(
        "-s",
        "--samplerate",
        dest="sampleRate",
        help="Generator sample rate",
        type=int,
        default=SAMPLERATE_STANDARD,
    )

    parser.add_argument(
        "--interp",
        help="Interpolation type",
        type=str,
        choices=[INTERP_NN, INTERP_LIN],
        default=INTERP_LIN,
    )

    main(parser.parse_args())
