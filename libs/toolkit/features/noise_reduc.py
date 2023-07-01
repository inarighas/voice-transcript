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
import noisereduce as nr
from .wav_transformer import WavTransformer
from .. import utils as utl


class NoiseReducer(WavTransformer):
    stationary: bool

    def __init__(
        self,
        input_folder: str,
        output_folder: Optional[str] = "noise_reduc",
        name: str = "Noise Reducer",
        processes: int = 1,
        stationary: bool = False,
        ):
        """_summary_

        Args:
            input_folder (str): _description_
            output_folder (str): _description_
            name (str, optional): _description_. Defaults to "Noise Reducer".
            processes (int, optional): _description_. Defaults to 4.
        """
        super().__init__(input_folder, output_folder, name=name, processes=processes)
        self.stationary: bool = stationary

    
    def transform(self, signal: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        """_summary_

        Args:
            signal (np.ndarray): _description_
            sr (int): _description_

        Returns:
            tuple[np.ndarray, int]: _description_
        """
        output_signal = nr.reduce_noise(
            y=signal, sr=sr, stationary=self.stationary, use_tqdm=False
        )
        
        return output_signal, sr
    
    def run_file(self, filename: str):
        """_summary_

        Args:
            filename (str): _description_
        """
        input_path = os.path.join(self.input_folder, filename)
        output_path = os.path.join(self.output_folder, filename)
        
        audio, sample_rate = utl.audioread(input_path)
        output_signal, _ = self.transform(audio, sample_rate)
        utl.audiowrite(output_path, output_signal, sample_rate)

        return
