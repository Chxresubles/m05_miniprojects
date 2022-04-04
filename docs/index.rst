.. Wine and Houses documentation master file, created by
   sphinx-quickstart on Mon Apr  4 11:03:41 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*******************************************
Welcome to Wine and Houses's documentation!
*******************************************

This repository holds 2 ML projects. Both projects wine_quality and boston_house_prices will perform Machine Learning tasks using 2 approaches:
Linear Regression and Regression Trees. Each project has its own dataset included.

Wine Quality & Boston House prices
##################################
This project is used to determe of the quality of a wine from their intrinsic properties.
P.S. For boston house price you can follow the same steps.

Description
***********
This project will determine the quality of a wine using 2 approaches: Linear Regression and Regression Trees.

Data sets
*********
Both the red and white wine data sets were retrieved from the UCI Machine Learning Repository that can be found here :
https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/ 

Prerequisites
*************
To run these scripts, you must have Python installed. You can install Python from here : https://www.python.org/

Getting started
***************
To start using these Python scripts locally, follow those steps : 

Installation
************************
* Clone the repo
   .. code-block::

          $ git clone https://github.com/Chxresubles/m05_miniprojects.git
          $ cd m05_miniprojects

* Go in the project directory
   .. code-block::

          $ cd wine_quality

* Create a Python virtual environment
   .. code-block::

          $ python -m venv wine_quality

* Activate the environment

   * For windows
      .. code-block::

          $ wine_quality\Scripts\activate.bat

   * For POSIX
      .. code-block::

          $ source wine_quality/bin/activate

* Install the required packages
   .. code-block::

          $ pip install -r requirements.txt

===========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
