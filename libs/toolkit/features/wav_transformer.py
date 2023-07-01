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
from abc import abstractmethod
import numpy as np
from joblib import Parallel, delayed
from typing import Optional
from tqdm import tqdm

# For multiproc uncomment
from multiprocessing import Pool


class WavTransformer:
    """Transforms a folder full of .wav into transformed .wav files
    Allows for parallel computation. To transform data, implement this
    class and add it to the WavPipeline.
    """
    name: str
    input_folder: str
    output_folder: str
    processes: int

    def __init__(
        self,
        input_folder: str,
        output_folder: Optional[str],
        name: str = "Transformer",
        processes: int = 4,
        ):
        """Instanciates WavTransformer object.

        Args:
            input_folder (str): _description_
            output_folder (str): _description_
            name (str, optional): _description_. Defaults to "Transformer".
                processes (int, optional): controls the number of threads.
                Defaults to 4.
        """
        self.name = name
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.processes = processes
        if output_folder is not None:
            os.makedirs(output_folder, exist_ok=True)

    @abstractmethod
    def transform(self, data: np.ndarray, sr: int):
        """Implement this function when you inherit the class."""
        pass

    @abstractmethod
    def run_file(self, filename: str):
        """Load data, run the transformer on it, then save the new data."""
        pass
    

    def run(self):
        """Runs the wave transformer on the input dataset in parallel.
        processes controls the number of threads"""
        logging.info(f"{self.name}")
        self.filenames = os.listdir(self.input_folder)
        Parallel(n_jobs=self.processes)(delayed(self.run_file)(f) for f in tqdm(self.filenames))
