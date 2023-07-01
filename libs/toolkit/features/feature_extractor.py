# -*- coding: utf-8 -*-
__author__ = "Ali Saghiran"
__copyright__ = "Copyright 2022, Resileyes Therapeutics - TraumaVoice project"
__credits__ = ["Ali Saghiran", "Older contributors"]
__version__ = "0.0.dev2"
__maintainer__ = "Ali Saghiran"
__email__ = "ali.saghiran@resileyes.com"
__status__ = "R&D"

from abc import abstractmethod
import os
import logging
from typing import Union, Optional
import opensmile
import numpy as np
import pandas as pd
from tqdm import tqdm

from .wav_transformer import WavTransformer
from .. import utils as utl


class FeatureExtractor(WavTransformer):
    """Abstract class for feature extracting modules"""

    def __init__(
        self,
        config,
        input_folder: str = "./",
        output_folder: str = "./",
        name: str = "FeatureExtractor",
    ):
        pass

    @abstractmethod
    def run(self, **kwargs):
        pass
    
    @abstractmethod
    def transform(self, **kwargs):
        pass


class DefaultOpenSmileFeatureExtractor(FeatureExtractor):
    config: str
    input_folder: str
    output_folder: Optional[str]
    feature_level: Union[str, opensmile.FeatureLevel]
    smile: opensmile.Smile
    
    def __init__(
        self,
        input_folder: str,
        config: str,
        output_file: Optional[str] = None,
        feature_level: Union[str,
                             opensmile.FeatureLevel
                             ] = opensmile.FeatureLevel.LowLevelDescriptors,
        **kwargs,
        ):
        """Apply the default configuration defined in OpenSmile
        (one of opensmile.FeatureSet).

        Args:
            input_folder (str): Folder where the audio files to process are stored.
            config (str): 1Name of the configuration in OpenSmile. Possible choices
                are  'ComParE_2016', 'GeMAPS', 'GeMAPSv01b', 'eGeMAPS',
                'eGeMAPSv01b' and 'eGeMAPSv02'.
            output_file (Optional[str], optional): Path to csv file where to store
                the extracted features. `None` values means no output file saving.
                Defaults to `None`.
            feature_level (Union[str, opensmile.FeatureLevel], optional):
                Name of the OpenSmile level where we wish to extract the features
                (see OpenSmile docs).
                Defaults to opensmile.FeatureLevel.LowLevelDescriptors.

        Raises:
            ValueError: Raises an error when the config is not found/available.
        """
        super(DefaultOpenSmileFeatureExtractor, self).__init__(config)
        self.config = config
        self.input_folder = input_folder
        self.output_file = output_file
        if self.output_file is not None:
            os.makedirs(output_file, exist_ok=True)  # type: ignore
        self.feature_level = feature_level
        feature_sets = {
            "ComParE_2016": opensmile.FeatureSet.ComParE_2016,
            "GeMAPS": opensmile.FeatureSet.GeMAPS,
            "GeMAPSv01b": opensmile.FeatureSet.GeMAPSv01b,
            "eGeMAPS": opensmile.FeatureSet.eGeMAPS,
            "eGeMAPSv01b": opensmile.FeatureSet.eGeMAPSv01b,
            "eGeMAPSv02": opensmile.FeatureSet.eGeMAPSv02,
        }
        if config not in feature_sets.keys():
            raise ValueError(
                f"config should be one of {feature_sets.keys()}. Got {config}"
            )
        
        self.feature_set = feature_sets[config]
        # default configuration file implemented in opensmile
        self.smile = opensmile.Smile(feature_set=self.feature_set,
                                     feature_level=self.feature_level,
                                     loglevel=0,
                                    )
        logging.debug(f"Selected feature sets {feature_sets[config]}")

    def run_file(self, filepath: str, filename: str) -> pd.DataFrame:
        """Run specific file and transform to a dataframe of extracted features.

        Args:
            file (str): Filename.

        Raises:
            ValueError: When file does not have wav extension.

        Returns:
            pd.DataFrame: dataframe of features with one or multiple rows in
                case of a sliding window processing.
        """
        if not filepath.endswith(".wav"):
            raise ValueError(f"File '{filepath}' has not the correct extension.")
        audiosig, sr = utl.audioread(filepath)
        df = self.transform(audiosig, sr)[0].assign(file=filename)
        return df
    
    def transform(self, signal: np.ndarray, sr: int):
        """Apply the default config file and returns a dataframe with all the features
        from the level `self.feature_level` of the config file. The features are NOT
        aggregated, different config files might return different shapes.
        
        Args:
            signal (np.ndarray): _description_
            sr (int): _description_

        Returns:
            pd.DataFrame: dataframe which rows are an aggregation type and columns
            a feature, for example `(mean, mfcc[0])`.

        """
        df = pd.DataFrame(
                self.smile
                .process_signal(signal, sr)
                .reset_index()
                )
        #.assign(file=filename)
        
        return df, None

    def run(self):
        """Transforms an input folder into a dataframe whcih index are the names of
        the files, and the columns are the aggregated feature values extracted using
        the OpenSmile configuration files from `config_folder`.

        Returns:
            pd.DataFrame: dataframe of features with one or multiple rows per file.
        """
        logging.info("Feature Extraction")
        filenames = [
            file for file in os.listdir(self.input_folder) if file.endswith(".wav")
        ]  # wav files in the folder
        filepaths = [os.path.join(self.input_folder, file) for file in filenames]

        logging.debug(f"Input folder path: {self.input_folder}.")
        logging.debug(f"{len(filenames)} files to process: {filenames}.")

        for i, f in enumerate(tqdm(filepaths)):
            df = self.run_file(f, filenames[i])
            if self.output_file is not None:  # save dataframe to csv file
                df.to_csv(
                    self.output_file + "/" + filenames[i][:-4] + "_features.csv",
                    index=False,
                )
        return