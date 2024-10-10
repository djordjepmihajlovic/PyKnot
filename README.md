# PyKnot for Machine Learned Knots & Knot Polynomial Coefficients.

### PyTorch MLKnotsProject code for TAPLAB 

PyKnot is a machine learning (ML) package with PyTorch and PyTorch.Lightning dependency to investigate the classification of mathematical knots. The package offers ML models to predict the classification of knots upto x crossings, and uses high level concept models to allow for interpretable learning.

The PyTorch version here is based off of an original Tensorflow code developed by TAPLAB.

## Running the code

To run a simple, class based knot prediction based on some input data (eg. SIGWRITHE (StA)) run main.py with the required arguments, for example:

```
    python main.py -modtyp NN -pred class -p 5Class -m train -d SIGWRITHE -t FFNN 
```
outputs the result of a neural network classification problem on the 5 class set (knots: 0_1, 3_1, 4_1, 5_1, 5_2) trained on segment to all (StA) data using a feedforward neural network

-pdct: takes in the prediction being made. "class" arguments correspond to classification of knots, whereas other invariants/codes can be specified. For example, "dowker" will return a training on dowker code prediction capability.
Options: class, dowker, v2, v3


## +  Models :

The following additions have been implemented into the existing framework in the new PyTorch version:

* Prediction (-pdct): new code capability to compute other prediction types other than standard class predicition

* Concept Bottle neck models (nn_concept_models.py): Torch.nn concept modules
    * ConceptNN(pl.LightningModule) 
    * OnlyConceptNN(pl.LightningModule)


