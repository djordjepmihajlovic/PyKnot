from argparse import ArgumentParser
import os
import numpy as np

from models.nn_models import *

def datafile_structure(dtype, knot, Nbeads, pers_len):
    """Returns datafile struct according to data type

    Args:
        dtype (str): datatype selected for the training/test
        knot (str): knot name
        Nbeads (int): number of beads in the knots
        pers_len (int): persistence length

    Raises:
        Exception: return expception if datatype not available
    
    """

    if dtype == "XYZ":
        header = None
        fname = os.path.join("XYZ", f"XYZ_{knot}.dat.nos") # XYZ/XYZ_{}.dat.nos basically
        select_cols = [0, 1, 2]
    
    elif dtype == "SIGWRITHE":
        header = True
        fname = os.path.join("SIGWRITHE", f"3DSignedWrithe_{knot}.dat.lp{pers_len}.dat.nos")
        select_cols = [2]

    elif dtype == "SIGWRITHElw10":
        header = True
        fname = os.path.join("SIGWRITHE", f"3DSignedWrithe_{knot}.dat.lp10.dat.nos")
        select_cols = [2]

    elif dtype == "UNSIGWRITHE":
        header = True
        fname = os.path.join("UNSIGWRITHE", f"3DUNSignedWrithe_{knot}.dat.lp{pers_len}.dat.nos")
        select_cols = [2]

    elif dtype == "DENSXSIGWRITHE":
        header = True
        fname = os.path.join("SIGWRITHEXLOCDENS", f"3DSignedWritheXLocDens_{knot}.dat.lp{pers_len}.dat.nos")
        select_cols = [4]

    elif dtype == "2DSIGWRITHE":
        header = True
        fname = os.path.join("SIGWRITHEMATRIX", f"3DSignedWritheMatrix_{knot}.dat.lp{pers_len}.dat")
        select_cols = np.arange(Nbeads)
    
    elif dtype == "2DSIGWRITHElw10":
        header = True
        fname = os.path.join("SIGWRITHEMATRIX", f"3DSignedWritheMatrix_{knot}.dat.lp10.dat")
        select_cols = np.arange(Nbeads)
    
    else:
        raise Exception("Datatype not available")
    
    return header, fname, select_cols


def generate_model(net, in_layer, out_layer, norm, predict):
    """Generate the model depeding on the net name

    Args:
        net (str): Net type
        in_layer (list): In layer size
        knots (list): List of the knots
        norm (bool): Bool activating batch normalisation layer

    Raises:
        Exception: Throw an exception of no model is found

    Returns:
        Pytorch model
    """

    # Loading different networks according to the chosen setup
    if net == "FFNN":
        model = setup_FFNN(in_layer, out_layer, opt="adam", norm=norm, loss="CEL", predict=predict)
    
    elif net == "RNN":
        model = setup_RNN(in_layer, out_layer, opt="adam", norm=norm, loss="MSE", predict=predict)

    elif net == "CNN":
        model = setup_CNN(in_layer, out_layer, opt="adam", norm=norm, loss="CEL", predict=predict)

    else:
        raise Exception("Network not available")

    return model



def get_knots(problem):
    """Return knot list based on the chosen problem

    Args:
        problem (str): Problem name

    Raises:
        Exception: Exception if problem is not available

    Returns:
        list: List of the knots to train over
    """

   # Loading knot list according to the problem
    if problem == "Conway":
        Knotind = ["0_1", "conway", "kt"]

    if problem == "test":
        Knotind = ["3_1", "4_1"]

    elif problem == "unknot":
        Knotind = ["0_1"]

    elif problem == "trefoil":
        Knotind = ["3_1"]

    elif problem == "0GlobalWrithe":
        Knotind = ["0_1", "4_1", "6_3"]

    elif problem == "5Class":
        Knotind = ["0_1", "3_1", "4_1", "5_1", "5_2"] # testing 5_2 unsupervised

    elif problem == "5Class5v6":
        Knotind = ["5_1", "5_2", "6_1", "6_2", "6_3"]

    elif problem == "5Class51v72":
        Knotind = ["5_1", "5_2", "7_1", "7_2", "7_3"]

    elif problem == "5Class910homfly":
        Knotind = ["9_1", "9_42", "10_1", "10_2", "10_71"]

    elif problem == "6Class":
        Knotind = ["0_1", "3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3"]

    elif problem == "7Class":
        Knotind = ["0_1", "3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3", "7_1", "7_2", "7_3", "7_4", "7_5", "7_6", "7_7"]

    elif problem == "8Class":
        Knotind = ["0_1", "3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3", "7_1", "7_2", "7_3", "7_4", "7_5", "7_6", "7_7", "8_1", "8_2", "8_3", "8_4", "8_5", "8_6", "8_7", "8_8", "8_9", "8_10", "8_11", "8_12", "8_13", "8_14", "8_15", "8_16", "8_17", "8_18", "8_19", "8_20", "8_21"]

    elif problem == "9Class":
        Knotind = ["0_1", "3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3", "7_1", "7_2", "7_3", "7_4", "7_5", "7_6", "7_7", "8_1", "8_2", "8_3", "8_4", "8_5", "8_6", "8_7", "8_8", "8_9", "8_10", "8_11", "8_12", "8_13", "8_14", "8_15", "8_16", "8_17", "8_18", "8_19", "8_20", "8_21", "9_1", "9_2", "9_3", "9_4", "9_5", "9_6", "9_7", "9_8", "9_9", "9_10", "9_11", "9_12", "9_13", "9_14", "9_15", "9_16", "9_17", "9_18", "9_19", "9_20", "9_21", "9_22", "9_23", "9_24", "9_25", "9_26", "9_27", "9_28", "9_29", "9_30", "9_31", "9_32", "9_33", "9_34", "9_35", "9_36", "9_37", "9_38", "9_39", "9_40", "9_41", "9_42", "9_43", "9_44", "9_45", "9_46", "9_47", "9_48", "9_49"]

    elif problem == "10Class":
        Knotind = ["0_1", "3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3", "7_1", "7_2", "7_3", "7_4", "7_5", "7_6", "7_7", "8_1", "8_2", "8_3", "8_4", "8_5", "8_6", "8_7", "8_8", "8_9", "8_10", "8_11", "8_12", "8_13", "8_14", "8_15", "8_16", "8_17", "8_18", "8_19", "8_20", "8_21", "9_1", "9_2", "9_3", "9_4", "9_5", "9_6", "9_7", "9_8", "9_9", "9_10", "9_11", "9_12", "9_13", "9_14", "9_15", "9_16", "9_17", "9_18", "9_19", "9_20", "9_21", "9_22", "9_23", "9_24", "9_25", "9_26", "9_27", "9_28", "9_29", "9_30", "9_31", "9_32", "9_33", "9_34", "9_35", "9_36", "9_37", "9_38", "9_39", "9_40", "9_41", "9_42", "9_43", "9_44", "9_45", "9_46", "9_47", "9_48", "9_49", "10_1", "10_2", "10_3", "10_4", "10_5", "10_6", "10_7", "10_8", "10_9", "10_10", "10_11", "10_12", "10_13", "10_14", "10_15", "10_16", "10_17", "10_18", "10_19", "10_20", "10_21", "10_22", "10_23", "10_24", "10_25", "10_26", "10_27", "10_28", "10_29", "10_30", "10_31", "10_32", "10_33", "10_34", "10_35", "10_36", "10_37", "10_38", "10_39", "10_40", "10_41", "10_42", "10_43", "10_44", "10_45", "10_46", "10_47", "10_48", "10_49", "10_50", "10_51", "10_52", "10_53", "10_54", "10_55", "10_56", "10_57", "10_58", "10_59", "10_60", "10_61", "10_62", "10_63", "10_64","10_65", "10_66", "10_67", "10_68", "10_69", "10_70", "10_71", "10_72", "10_73", "10_74", "10_75", "10_76", "10_77", "10_78", "10_79", "10_80", "10_81", "10_82", "10_83", "10_84", "10_85", "10_86", "10_87", "10_88", "10_89", "10_90", "10_91", "10_92", "10_93", "10_94", "10_95", "10_96", "10_97", "10_98", "10_99", "10_100", "10_101", "10_102", "10_103", "10_104", "10_105", "10_106", "10_107", "10_108", "10_109", "10_110", "10_111", "10_112", "10_113", "10_114", "10_115", "10_116", "10_117", "10_118", "10_119", "10_120", "10_121", "10_122", "10_123", "10_124", "10_125", "10_126", "10_127", "10_128", "10_129", "10_130", "10_131", "10_132", "10_133", "10_134", "10_135", "10_136", "10_137", "10_138", "10_139", "10_140", "10_141", "10_142", "10_143", "10_144", "10_145", "10_146", "10_147", "10_148", "10_149", "10_150", "10_151", "10_152", "10_153", "10_154", "10_155", "10_156", "10_157", "10_158", "10_159", "10_160", "10_161", "10_162", "10_163", "10_164", "10_165"]

    elif problem == "SQRGRN8":
        Knotind = ["3_1_3_1", "3_1-3_1", "8_20"]  # square knot, granny knot, 8_20

    elif problem == "sameVASSILIEV":
        Knotind = ["5_1", "7_2", "3_1_3_1", "3_1-3_1", "8_20"]

    else:
        raise Exception("Problem not available")

    return Knotind

def get_params():
    """Receive user-input of training parameters via the command Line interface (CLI) and Python library argparse.
    Default values are provided if no input is specified.

    Returns:
        args: Values defining the knot parameters.
    """
    par = ArgumentParser()

    par.add_argument("-p", "--problem", type=str, default="test", help="Options: Conway, 5Class, SQRGRN8, 10Crossings, ...")
    par.add_argument("-d", "--datatype", type=str, default="SIGWRITHE", help="Options: XYZ, SIGWRITHE, UNSIGWRITHE, DENSXSIGWRITHE, 2DSIGWRITHE")
    par.add_argument("-a", "--adjacent", type=bool, default=False, help="Flag to use adjacent datatype from XYZ (deprecated)")
    par.add_argument("-n", "--normalised", type=bool, default=False, help="Flag to use normalised version of datatype")
    par.add_argument("-nb", "--nbeads", type=str, default="100", help="Number of beads of the input files")
    par.add_argument("-t", "--network", type=str, default="FFNN", help="Type of neural network: FFNN, RNN, LocaliseFFNN, LocaliseRNN, ...")
    par.add_argument("-e", "--epochs", type=int, default=1000, help="Set the number of training epochs")
    par.add_argument("-m", "--mode", type=str, default="train", help="Mode: train, test, generate")
    par.add_argument("-ldb", "--len_db", type=int, default=100000, help="Database size for each of the classes")
    par.add_argument("-bs", "--b_size", type=int, default=256, help="Batch size") 
    par.add_argument("-mkndir", "--master_knots_dir", type=str, default="/Users/s1910360/Desktop/ML for Knot Theory/sample_data", help="Batch size")
    ## below is location on cluster..
    # par.add_argument("-mkndir", "--master_knots_dir", type=str, default="/storage/cmstore04/projects/TAPLabKnotsDatabase/knots_database/", help="Batch size")
    par.add_argument("-lp", "--pers_len", type=int, default=10, help="Persistence Length")
    par.add_argument("-pdct", "--predictor", type=str, default="class", help="Options: class, dual")
    par.add_argument("-modtyp", "--model_type", type=str, default="NN", help="Options: NN, DT, LogR, LinR")

    args = par.parse_args()

    return args

