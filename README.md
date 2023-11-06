PyTorch MLKnotsProject code by TAPLAB

+Additions:

The following additions have been implemented into the existing framework in the new PyTorch version:

    Prediction (-pred): new code capability to compute other prediction types other than standard class predicition, such as XYZ coordinate to SIGWRITHE predicitions (std, dual)

    Further SHAP analysis


-Removals:

Currently, not implemented into the PyTorch version:

    2DSIGWRITHE (Data): -
    DENSXSIGWRITHE (Data): -
    RNN2, 2b (network): -
    FFNN2 (network): -
    localise_* (network): -
    randFOR (network): -


=Difference:

The following methods are implemented differently to the TensorFlow base code and are subject to change:

    Currently no separate train/test functionality; testing occurs immediately after training


!Current project work-points:

    Implementation of XYZ to StA predicition & vice versa.

    Generative models.

    Model analysis.





