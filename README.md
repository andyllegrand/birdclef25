Train - simple resnet finetune, computes mel specs in real time so runs very slow. Uses first 10 seconds of each file. Reaches about 55% accuracy score
precompute melspecs - computes and saves all melspecs for dataset. Divides each file into 10s clips and makes melspec for each.
train_with_pc - finetunes resnet with precomputed melspecs. Much faster. Achieves 65% accuracy
