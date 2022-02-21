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

import numpy


# ============================================================================================================
# Constants
# ============================================================================================================


# ============================================================================================================
# Useful functions
# ============================================================================================================

def estimate_norm(data):
  return data.mean(axis=0), data.std(axis=0, ddof=1)


def normalize(data, norm):
  return numpy.array([(k - norm[0]) / norm[1] for k in data])


def min_max_scaling(data):
  return None


def z_norm(data):
  return None


def min_max_scaling_poly(data, polynoms):
  return None


def z_norm_ poly(data, polynoms):
  return None