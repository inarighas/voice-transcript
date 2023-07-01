# -*- coding: utf-8 -*-
__author__ = "Ali Saghiran"
__copyright__ = "Copyright 2022, Resileyes Therapeutics - TraumaVoice project"
__credits__ = ["Ali Saghiran", "Older contributors"]
__version__ = "0.0.dev2"
__maintainer__ = "Ali Saghiran"
__email__ = "ali.saghiran@resileyes.com"
__status__ = "R&D"

from sklearn.pipeline import Pipeline


class WavPipeline(Pipeline):
    """Runs a few WavTransformers."""

    def __init__(self, steps: list[tuple]):
        self.steps = steps

    def run(self):
        for _, transformer in self.steps:
            transformer.run()

    def transform(self, X, sr):
        """Transform the data, and apply `transform` with the final transformer.
        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final transformer that calls
        `transform` method. Only valid if the final tranformer
        implements `transform`.

        Args:
            X (np.ndarray): Data to transform. Must fulfill input requirements of first step
            of the pipeline.

        Returns:
            X_tmp (np.ndarray): Transformed data.
        """
        X_tmp = X
        sr_tmp = sr
        for _, transformer in self.steps:
            X_tmp, sr_tmp = transformer.transform(X_tmp, sr_tmp)
        return X_tmp