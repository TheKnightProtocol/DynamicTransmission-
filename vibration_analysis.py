import pandas as pd

class VibrationAnalysis:

    @staticmethod
    def rms(signal):

        return (signal**2).mean()**0.5

    @staticmethod
    def peak(signal):

        return signal.max()

    @staticmethod
    def crest_factor(signal):

        return signal.max()/((signal**2).mean()**0.5)
