# -*- coding: utf-8 -*-
__author__ = "Ali Saghiran"
__copyright__ = "Copyright 2022, Resileyes Therapeutics - TraumaVoice project"
__credits__ = ["Ali Saghiran", "Older contributors"]
__version__ = "0.0.dev2"
__maintainer__ = "Ali Saghiran"
__email__ = "ali.saghiran@resileyes.com"
__status__ = "R&D"

import os
from typing import Optional
import numpy as np
from librosa import resample
from .wav_transformer import WavTransformer
from .. import utils as utl


class Resampler(WavTransformer):
    """Transforms .wav files into the desired sample_rate"""

    def __init__(
        self,
        input_folder: str,
        output_folder: Optional[str] = "data_resampled",
        name: str = "Resampler",
        processes: int = 4,
        desired_sr: int = 8000,
    ):
        super().__init__(input_folder, output_folder, name=name, processes=processes)
        self.desired_sr = desired_sr

    def transform(self, data: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        """Perform the resampling tranformation.

        Args:
            data (np.ndarray): Input array (discrete audio signal)
            sr (int): Original sampling rate.

        Returns:
            np.ndarray: Resampled signal.
            int: Desired sample rate.
        """
        return resample(data, orig_sr=sr, target_sr=self.desired_sr), self.desired_sr

    def run_file(self, filename: str):
        """Reads the input file, perform the transformation and writes the iutput file.

        Args:
            filename (str): Input filename (or path).
        """
        input_path = os.path.join(self.input_folder, filename)
        output_path = os.path.join(self.output_folder, filename)
        data, sr = utl.audioread(input_path)
        data_resampled, _ = self.transform(data, sr)
        utl.audiowrite(output_path, data_resampled, self.desired_sr)
