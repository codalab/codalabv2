#!/usr/bin/env python

# Scoring program for the See4C challenge
# Isabelle Guyon, January 2016

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# ISABELLE GUYON, SEE4C, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRINGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS.
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL,
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS,
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE.

# Some libraries and options
import os

from sys import argv
import numpy as np
import yaml
from metric import scoring_function, score_name
from libscores import mkdir, mvmean, swrite
from libscores import show_platform, show_io, show_version
from data_manager import DataManager
from data_io import vprint

# Default I/O directories:
root_dir = "../"
default_solution_dir = root_dir + "sample_data"
default_prediction_dir = root_dir + "sample_results"
default_score_dir = root_dir + "scoring_output"

# Debug flag 0: no debug, 1: show all scores, 2: chaeting results (use ground truth)
debug_mode = 1
verbose = True

# Constant used for a missing score
missing_score = 0.999999

# Version number
scoring_version = 1.0

# Extension of the files
ext = '.csv'

# =============================== MAIN ========================================

if __name__=="__main__":

    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv) == 1:  # Use the default data directories if no arguments are provided
        solution_dir = default_solution_dir
        prediction_dir = default_prediction_dir
        score_dir = default_score_dir
    elif len(argv) == 3: # The current default configuration of Codalab
        solution_dir = os.path.join(argv[1], 'ref')
        prediction_dir = os.path.join(argv[1], 'res')
        score_dir = argv[2]
    elif len(argv) == 4:
        solution_dir = argv[1]
        prediction_dir = argv[2]
        score_dir = argv[3]
    else:
        swrite('\n*** WRONG NUMBER OF ARGUMENTS ***\n\n')
        exit(1)

    # Create the output directory, if it does not already exist and open output files
    mkdir(score_dir)
    score_file = open(os.path.join(score_dir, 'scores.txt'), 'w')
    html_file = open(os.path.join(score_dir, 'scores.html'), 'w')
    html_file.write('<pre>')

    # Get all the solution files from the solution directory
    vprint( verbose,  "========= Reading solution data ==========")
    Dsol = DataManager(datatype="input", verbose=verbose)
    Dsol.loadData(solution_dir)
    vprint( verbose, Dsol)
    # Get the predictions
    vprint( verbose,  "========= Reading prediction data ==========")
    Dpred = DataManager(datatype="output", verbose=verbose)
    Dpred.reloadData(os.path.join(prediction_dir, "prediction"), format="pickle")
    vprint( verbose, Dpred)

    # Score results
    max_steps = Dsol.t.shape[0]-Dsol.t0
    score = np.zeros(max_steps)
    for step_num in range(max_steps):
        vprint( verbose,  "step {}".format(step_num))
        # Get next solutions for the given horizon
        Xsol, tsol = Dsol.getFutureOutcome()
        # Get next predictions for the given horizon
        Xpred, tpred = Dpred.getPredictions()
        # Check time stamps
        if sum(tsol-tpred)!=0:
             vprint( verbose,  "[-] Error time stamps differ tsol={}, tpred={}".format(tsol, tpred))
             exit(0)
        if True: #try:
             # Compute the score prescribed by the metric file
             score[step_num] = scoring_function(Xsol, Xpred)
             vprint( verbose,  "======= Step {}: {}={:0.12f} =======".format(step_num, score_name, score[step_num]) )
             html_file.write("======= Step {}: {}={:0.12f} =======\n".format(step_num, score_name, score[step_num]) )
        else: #except Exception as inst:
            score[step_num] = np.nan
            print("======= Step {}: ERROR =======")
            html_file.write("======= Step {}: ERROR =======\n")
    # End loop over solutions

    # Average scores:
    prediction_score = np.mean(score)
    print("*** prediction_score (average {}) = {:0.12f} ***".format(score_name, prediction_score))
    html_file.write("*** prediction_score (average {}) = {:0.12f} ***".format(score_name, prediction_score))
    # Write score corresponding to selected task and metric to the output file
    if np.isnan(prediction_score): prediction_score=missing_score
    score_file.write("prediction_score: {:0.12f}\n".format(prediction_score))

    # Read the execution time and add it to the scores:
    try:
        metadata = yaml.load(open(os.path.join(prediction_dir,'metadata'), 'r'))
        score_file.write("Duration: {:0.6f}\n" % metadata['elapsedTime'])
    except:
        score_file.write("Duration: 0\n")

	html_file.write('</pre>')

    html_file.close()
    score_file.close()

    # Lots of debug stuff
    if debug_mode>1:
        swrite('\n*** SCORING PROGRAM: PLATFORM SPECIFICATIONS ***\n\n')
        show_platform()
        show_io(prediction_dir, score_dir)
        show_version(scoring_version)

    #exit(0)



