# MixedTrails

This repository contains the code used to produce the results from the following paper:

>Martin Becker, Florian Lemmerich, Philipp Singer, Markus Strohmaier, Andreas Hotho  
**MixedTrails: Bayesian Hypotheses Comparison on Heterogeneous Sequential Data**  
[Preprint available on ArXiv](https://arxiv.org/abs/1612.07612).

The experiments are conducted on 
* a **toy** example about soccer strategies, 
* **synthetic** navigation paths on random graphs,
* navigation from the game **Wikispeedia**, and
* photowalking trails from the social photo-sharing platform **Flickr**.

## Datasets

For the **toy** example and the **synthetic** experiments no additional data is needed.

### Wikispeedia
The required Wikispeedia data can be downloaded from
> https://snap.stanford.edu/data/wikispeedia.html

These two files are needed:
* `wikispeedia_paths-and-graph.tar.gz`
* `wikispeedia_articles_plaintext.tar.gz`

Directly extract both files into the `wikispeedia` folder. This should result in two folders:
* `wikispeedia_paths-and-graph`
* `plaintext_articles`


### Flickr
For the Flickr data, please contact Martin Becker via e-mail:
>becker@informatik.uni-wuerzburg.de

## Run Experiments

For running the experiments you need `python3`. 
The requirements file for `pip` can be found in the root folder of this project (`requirements.txt`):
```
matplotlib==1.5.3
numpy==1.11.2
scipy==0.18.1
scikit_learn==0.18.1
```

After setting up your workspace, running the actual experiments is straight forward.
For the **toy** example, the **synthetic** data as well as the **Flickr** experiments
the corresponding Jupyter notebooks (`exp-toy-offense.ipynb`, `exp-synthetic.ipynb`, `exp-flickr.ipynb` respectively) 
are self contained.
You can simply run them and have a look at the results.

For the **Wikispeedia** experiments you first need to run two scripts before 
the results can be visualized in the corresponding notebook (`exp-wikispeedia-visualization.ipynb`):
```
python exp-wikispeedia-prepare-data.py
python exp-wikispeedia.py
```

## Contact
>Martin Becker  
DMIR Group, University of Würzburg  
becker@informatik.uni-wuerzburg.de
