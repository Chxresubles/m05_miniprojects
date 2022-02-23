#!/usr/bin/env python

"""
Preprocessing funtions for the wine quality ML project.

This file was given with the course materials for the ex3
of the module M05 of the AI master at Idiap. Small
modifications were done to fit the file with our needs
for this mini-project.
"""


# ============================================================================================================
# Imports
# ============================================================================================================

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ============================================================================================================
# Constants
# ============================================================================================================


# ============================================================================================================
# Useful functions
# ============================================================================================================

def estimate_norm(data):
  """Estimate the mean and the std of the data."""
  return data.mean(axis=0), data.std(axis=0, ddof=1)


def normalize(data, norm):
  """Normalize the data using the given mean and std."""
  return np.array([(k - norm[0]) / norm[1] for k in data])


def min_max_scaling(data):
  """Min Max Scaling of the data."""
  scaler = MinMaxScaler()
  return scaler.fit_transform(data)


def z_norm(data):
  """Z normalisation of the data."""
  scaler = StandardScaler()
  return scaler.fit_transform(data)


def min_max_scaling_poly(data, polynoms):
  """Min Max Scaling of the data using the given polynomial function."""
  return None


def z_norm_ poly(data, polynoms):
  """Z normalisation of the data using the given polynomial function."""
  return None