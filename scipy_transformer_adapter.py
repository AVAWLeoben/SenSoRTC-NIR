# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 12:42:04 2026

@author: GKoinig
"""

from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin

class SavGolTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, window_length=15, polyorder=2, deriv=1):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return savgol_filter(
            X,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            axis=1,
            mode="interp",
        )