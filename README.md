# Holographic Embeddings of Knowledge Graphs

This repository holds the code for experiments in the paper 

```
Holographic Embeddings of Knowledge Graphs
Maximilian Nickel, Lorenzo Rosasco, Tomaso Poggio, AAAI 2016.
```

**plus** aditional modifications by the team A. Diaz, H. Naranjo and R. Paganini for the [Geometric Data Analysis](https://www.jeanfeydy.com/Teaching/index.html) project in the master [_Mathématiques Vision Apprentissage_](https://www.master-mva.com/) (ENS Paris-Saclay).
Concretely, these modifications include:
- Experimental code for obtaining meaningful representations to unseen entities ([`research/embedding_extrapolation_experiment.ipynb`](research/embedding_extrapolation_experiment.ipynb))
- Automatic conversion of datasets to an accessible format for the repository ([`research/convert_...`](research/)).
- Logging and tensorboard generation for training experiments.

## Install 

To run the experiments, first install [scikit-kge](https://github.com/mnick/scikit-kge),
An open-source python library to compute various knowledge graph embeddings including

- Holographic Embeddings (HolE)
- RESCAL
- TransE
- TransR
- ER-MLP

After `scikit-kge` is installed, simply clone this repository via 

```
git clone git@github.com:mnick/holographic-embeddings.git
```

and run the experiments as detailed in the next section

## Experiments 

The repository holds scripts of the form 

```
run_<model>_<dataset>.sh
```

which runs the experiments for `dataset` with the best parameters for `model`.

The full code for the experiments can be found in the `kg` and `countries` subfolders. The python scripts in these subfolders should be easy to use for grid search.
