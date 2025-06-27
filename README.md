**Overview**
Our training code for the birdCLEF 2025 challenge. Our inference code can be found [here](https://www.kaggle.com/code/hanchenbai/birdclef2025-inference-notebook). We achieved a final score of .804 ROC-AUC, placing 905th in the competition. Our approach used mel spectrograms (a visual representation for sounds) combned with CNNs.

**File Desciptions**
Final report - PDF containing our final report for CSE493S

Clean Data - Removes loudest 20% of data and drops duplicates as suggested in BirdCLEF2024 winning solution. 

Filter Dataset - Used to set up experiment 3 of our paper. Drops all data samples which the specified ensemble fails to accuratly predict.

Grid Search - Performs a grid search across hyperparameters and records the best performing config.

Precompute Specs - Computes and saves all melspecs for dataset. Divides each file into 10s clips and computes and saves a melspec for each.

Train - Original training NB. Simple resnet finetune, computes mel specs in real time, runs very slow. Uses first 10 seconds of each file. Reaches about 55% accuracy score

Train Ensemble - Trains an ensemble of models.

Train with PC - Training notebook for models which utilize precomputed mel spectrograms. Runs signicantly faster than train.py
