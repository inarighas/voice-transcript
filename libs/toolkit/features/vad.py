# -*- coding: utf-8 -*-
__author__ = "Ali Saghiran"
__copyright__ = "Copyright 2022, Resileyes Therapeutics - TraumaVoice project"
__credits__ = ["Ali Saghiran", "Older contributors"]
__version__ = "0.0.dev2"
__maintainer__ = "Ali Saghiran"
__email__ = "ali.saghiran@resileyes.com"
__status__ = "R&D"

import os
import logging
import collections
import webrtcvad
from typing import Generator, Optional, Union, cast
import simplejson as sj
import numpy as np
from .. import utils as utl
from .wav_transformer import WavTransformer


class Frame(object):
    """Represents a frame of audio data."""

    content: bytes
    timestamp: float
    duration: float

    def __init__(self, content: bytes, timestamp: float, duration: float):
        """Instantiates Frame object.

        Args:
            content (bytes): Byte array of the corresponding audio frame.
            timestamp (float): Timestamp (milliseconds) of the beginning
                of the frame.
            duration (float): Duration in milliseconds of the frame.
        """
        self.content = content
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(
    frame_duration_ms: int, audio: np.ndarray, sample_rate: float
) -> Generator[Frame, None, None]:
    """Generates fixed size audio frames from audio data.

    Args:
        frame_duration_ms (int): Desired frame duration in millisecond.
        audio (np.ndarray): Array of audio samples.
        sample_rate (float): Sampling rate / sampling frequency.

    Yields:
        Generator[Frame, None, None]: Frame object of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0))

    duration = frame_duration_ms / 1000.0
    offset = 0
    timestamp = 0.0
    while offset + n < len(audio):
        yield Frame(utl.float_to_byte(audio[offset : offset + n]), timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(
    sample_rate: int,
    frame_duration_ms: int,
    padding_duration_ms: int,
    vad: webrtcvad.Vad,
    frames: list[Frame],
) -> Generator[tuple[bytes, list], None, None]:
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Args:
        sample_rate (int): The audio sample rate, in Hz.
        frame_duration_ms (int): The frame duration in milliseconds.
        padding_duration_ms (int): The amount to pad the window, in
            milliseconds.
        vad (webrtcvad.Vad): An instance of `webrtcvad.Vad`.
        frames (Generator): a source of audio frames, a generator of
            `Frame` instances.


    Yields:
        Iterator[tuple[bytes, list]]: A generator that yields PCM audio
            data and a list of corresponding beginning & ending timestamps
            in seconds.
    """

    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.content, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the TRIGGERED state.
            if num_voiced > (0.9 * num_padding_frames):
                triggered = True
                # We want to yield all the audio we see from now until we are
                # NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data and add
            # it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are unvoiced,
            # then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * num_padding_frames:
                triggered = False
                yield b"".join([f.content for f in voiced_frames]), [
                    voiced_frames[0].timestamp,
                    voiced_frames[-1].timestamp,
                ]
                ring_buffer.clear()
                voiced_frames = []
    # If we have any leftover voiced audio when we run out of input, yield it.
    if voiced_frames:
        yield b"".join([f.content for f in voiced_frames]), [
            voiced_frames[0].timestamp,
            voiced_frames[-1].timestamp,
        ]


class VAD(WavTransformer):
    """Filters out the parts in a .wav file where no one is speaking"""

    def __init__(
        self,
        input_folder: str,
        output_folder: Optional[str] = "vad",
        name: str = "VAD",
        processes: int = 4,
        json: bool = False,
        write: bool = True,
        aggressivity: int = 2,
    ):
        """Instanciates VAD object.

        Args:
            input_folder (str): Audio input folder
            output_folder (str, optional): Defaults to "vad".
            name (str, optional): Defaults to "VAD".
            processes (int, optional): Defaults to 4.
            aggressivity (int, optional): Aggressivity goes from 0 to 3.
                Defaults to 2.
            json (bool, optional): If True, writes in the output folder
                a json file containing timestamps of voiced frames.
                Defaults to False.
            write (bool, optional): If True, writes in the output folder
                a wavfile os the processed file. Defaults to True.
        """
        super().__init__(input_folder, output_folder, name=name, processes=processes)
        self.aggressivity = aggressivity
        self.json = json
        self.write_output = write

    def transform(self, 
                  signal: np.ndarray,
                  sr: int,
                  timestamps=False
                  ) -> Union[tuple[np.ndarray, int, list], tuple[np.ndarray, int]]:
        
        vad = webrtcvad.Vad(self.aggressivity)
        frames = frame_generator(30, signal, sr)
        frames = list(frames)

        segments = vad_collector(sr, 30, 300, vad, frames)

        concat_audio, concat_timestamps = [], []
        for segment in segments:
            s, t = segment
            concat_audio.append(s)
            concat_timestamps.append(t)

        logging.debug("Running VAD and saving output file.")
        joined =  utl.byte_to_float(b"".join(concat_audio))
        
        if timestamps:
            return joined, sr, concat_timestamps
        
        return joined, sr

    
    def run_file(self, filename: str):
        """Load data from audio file, run the silence detector and remover on it,
        then save the new data.

        Args:
            filename (str): input audio file name.
        """
        input_path = os.path.join(self.input_folder, filename)
        logging.debug(f"Loading file: '{input_path}'.")
        audio, sample_rate = utl.audioread(input_path)
        transformed = self.transform(audio, sample_rate, timestamps=self.json)
        
        tmstmp = None
        if self.json:
            _, _, tmstmp = cast(tuple[np.ndarray, int, list], transformed)
            json_path = os.path.join(self.output_folder, "json/")
            os.makedirs(json_path, exist_ok=True)

            logging.debug(f"Writing timestamp JSON file for {filename}.")
            with open(os.path.join(json_path, filename[:-4] + ".json"), "w") as f:
                sj.dump(tmstmp, f)
        
        if self.write_output:
            output_path = os.path.join(self.output_folder, filename)
            utl.audiowrite(output_path, transformed[0], transformed[1])

        return
