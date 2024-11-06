# How to run:
# 
# main.py [dataset] [featureExtract] [algorithm] [seed]

import sys
import os
import config
import pickle as pickle

from runExperiment import runExperiment

if __name__ == "__main__":
    
    # load sys.arg scripts
    if(len(sys.argv) != 5):
        print("Usage: main.py [dataset] [featureExtract] [algorithm] [seed]")
        sys.exit(1)
    
    # check if dataset is valid
    if(not(sys.argv[1] in config.VALID_DATASETS)):
        print("Dataset is not valid")
        sys.exit(1)

    # check if featureExtract is valid 
    if(not(sys.argv[2] in config.VALID_FEATURES)):
        print("Feature Type is not valid")
        sys.exit(1)

    # check if algorithm is valid
    if(not(sys.argv[3])in config.VALID_ALGORITHMS):
        print("Algorithm is not valid")
        sys.exit(1)

    # check if seed is valid
    if(not(sys.argv[4].isdigit()) and int(sys.argv[4]) < 0):
        print("Seed is not valid")
        sys.exit(1)

    # ----------------
    # For debugging
    # ----------------
        
    # dataset = "user_reviews_g1"
    # algorithm = "KNN"
    # featureExtract = "TFIDF"
    # seed = 0
        
    dataset = sys.argv[1]
    featureExtract = sys.argv[2]
    algorithm = sys.argv[3]
    seed = int(sys.argv[4])

    output_file = "output/results_" + dataset + "_" + featureExtract + "_" + algorithm + "_seed_" + str(seed) + ".pkl"
    print(output_file)

    # check if output file exists
    if(os.path.exists(output_file)):
        print("\n - Output file already exists. Not running again!")
        sys.exit(1)

    # othersise, we can run experiments
    results = runExperiment(dataset = dataset, featureExtract = featureExtract, 
                            algorithm = algorithm, seed = seed)
    
    #create output dir if not exists
    os.makedirs("output", exist_ok=True)

    # save results
    print(" - Saving Results")

    pickle.dump(results, file = open(output_file, "wb"))
   
    print(" - Finished :)")