.. Wine and Houses documentation master file

**********************************************
Welcome to the m05_miniprojects documentation!
**********************************************

This repository holds 2 ML projects. Both projects wine_quality and boston_house_prices will perform Machine Learning tasks using 2 approaches:
Linear Regression and Regression Trees. Each project has its own dataset included.

Wine Quality & Boston House prices
##################################

Description
***********
The wine_quality project is used to determine of the quality of a wine from their intrinsic properties.

The boston_house_prices project is used to predict the price of a house in Boston from their features.

Data sets
*********
Both the red and white wine data sets were retrieved from the UCI Machine Learning Repository that can be found here :

https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/

The Boston houses data sets were retrieved from the UCI Machine Learning Repository that can be found here :

https://archive.ics.uci.edu/ml/machine-learning-databases/housing/

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

          $ python -m venv m05_group3

* Activate the environment

   * For windows
      .. code-block::

          $ m05_group3\Scripts\activate.bat

   * For POSIX
      .. code-block::

          $ source m05_group3/bin/activate

* Install the required packages
   .. code-block::

          $ pip install -r requirements.txt

* Install the m05_miniprojects package
   .. code-block::

          $ pip install .

* Run the ML algorithm
   .. code-block::

          $ wine_quality
          $ boston_house_prices

* Arguments and examples can be found by running:
   .. code-block::

          $ wine_quality --help
          $ boston_house_prices --help


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
