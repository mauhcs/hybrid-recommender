A recommendation system by Mau Hernandes, Ph.D. - September 2020

# The Problem: 

Item Recommendation System for users of a (web) app. 

# The setup
Given the artificial files `item_history.tsv`, `user_master.tsv` and `target_users.tsv` (see 'how to download' section), to build a recommendation system that maximizes the accuracy of the recommendation but with certain level of variaty (entropy) on the selection.

This repo contains 3 main componets: 1) A PDF report with the key concepts and ideas for this project, 2) Jupyter notebooks going through the steps of building the system and 3) a few python files with functions and classes used for some of the notebooks. 

The final recommendation model is in the notebook called `5. Model`


## Suggested reading order:

First check the pdf for an overview of the system. Start with the the abstract and introduction but them feel free to jump to the last section called `Our Results` with numerical results with the performance of the system. Then open the last Jupyter Notebook (The Model) to see the Recommender System in Action.


## System Requirements

> spark 3.0.0
> python 3.8
- tested on Ubuntu 20

A quick tutorial on how to install Spark on Ubuntu: https://medium.com/solving-the-human-problem/installing-spark-on-ubuntu-20-on-digital-ocean-in-2020-a7e4b5b65ffb

## Folder description

### Notebooks:

  1. Cleaner: for cleaning the data for training and testing used by the other notebooks
  2. Feature Exploration: A few histograms of the distribution of some of the features in the `user_master.tsv` file.
  3. ALS Training: An Alternating Least Squares training notebook for Collaborative Filtering. Including gridsearch and evaluation with nDCG.
  3. kMeans:  A Kmeans training notebook for Contente Based Filtering. Including gridsearch and evaluation with nDCG.
  4. Evaluation: A notebook for quick trying and evaluating different models. 
  5. Model: The notebook with the recommendation system proposed by this project. A mix of ALS + Kmeans. 

### PDF Report:

  The pdf file is an overview of the methods and technologies applied in the development of the recommendation system. It includes some mathematical discussions of key concepts, some plots and tables from our benchmark tests and a extensive description and diagrams of how our system works.

### Python files:

  1. model:
    - als_trainer.py: Convenience functions for cleaning the data and training an ALS model
    - kmeans_trainer.py: Convenience functions for cleaning the data and fitting a k-means cluster.
  2. utils:
    - evaluate.py: File containing our `evaluate` function to benchmark different models
    - make_Y.py:   File for cleaning the data to make training and testing data
    - metrics.py:  File containg functions to calculate the nDCG metrics.

### Data Folder:

  1. data: Contain all the data (`.tsv` files) that the different functions and methods read and write to.
  2. models: contains the pre trained/fitted models for the recommendation system. (Actually I omitted the model from the repo because the final version had 500MB+)


  