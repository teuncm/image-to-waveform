# Image To Waveform

This program is designed to superimpose an image mask on a waveform. Images should be drawn on a black (RGB(0, 0, 0)) canvas. Inspired by [this YouTube video](https://www.youtube.com/watch?v=qeUAHHPt-LY).

## Requirements
Requires `numpy, pillow, pysoundfile`, e.g. through:
```
pip install numpy Pillow PySoundfile
```

## Basic usage
```
python3 image-to-waveform.py in.png out.wav
```

## Parameters
```
-i INAUDIO, --inAudio INAUDIO
                    Input audio filename
-d DURATION, --duration DURATION
                    Length of generated audio in seconds
-f FREQUENCY, --frequency FREQUENCY
                    Generator frequency
-s SAMPLERATE, --samplerate SAMPLERATE
                    Generator sample rate
--interp {nn,lin}     Interpolation type
```


## How it works
The program transforms image pixels into audio offsets and amplitudes. The resulting values are then interpolated to smooth the transitions of the waveform. As the time dimension in audio only increases, the image has to have certain properties in order to be fully representable in the audio (there should be no discontinuities in the y-axis).

The default waveform is a cosine wave at maximum amplitude with the [Nyquist frequency](https://en.wikipedia.org/wiki/Nyquist_frequency). This frequency is chosen such that the resulting samples continuously alternate between -1 and +1, and thus provide most visible body to the resulting waveform. Of course, the user is free to choose any other frequency or even to load their own waveform. The default result is inaudible for the average human (22050 Hz), but should be readily visible through e.g. [Audacity](https://www.audacityteam.org/).
