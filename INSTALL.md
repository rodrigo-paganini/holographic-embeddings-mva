# Installation guide


To install the repo, first perform:

```
git clone https://github.com/mnick/holographic-embeddings.git
```

Install scikit kge. Pip wheels may not work, in which case you need to do git clone and extract the `skge` folder to the root directory.

```
git clone https://github.com/mnick/scikit-kge.git
cp -r scikit-kge/skge .
```

You can also create an environment and install the other dependencies:

```
pip install numpy scikit-learn scipy nose ipykernel
```

or alternatively:

```
pip install -r requirements.txt
```

## Datasets

Original downloads:
- [OpenBioLink](https://openbiolink.github.io/dataset/)

Dev requirements to handle these datasets:

```
pip install openbiolink
```