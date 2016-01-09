1. The "Code" folder contains the source code for CCLDA, CCLDA, class-slda (ssLDA in the paper), and LDA. Compile each program in a Linux-based system by typing "make".

The class-slda code is obtained from https://github.com/Blei-Lab/class-slda and then modified to handle both labeled and unlabeled training documents.

2. The "data" folder contains the training and test data sets we used in our experiments in the paper. In most cases, it also contains necessary python scripts to download the data, do the required pre-processing, and split the data into training and test sets. For instance, "prepare_ag_data.py" will download the raw corpus and generate the same training and test sets used in the paper.

3. The actual experiments are in the "Experiments" folder. For each dataset and every label proportion, we have a folder in the path: "Dataset/Method/propid/rep/" where Dataset = {20ng, ag, dbpedia}, Method = {mcctm, cclda, sslda, sslda-trans}, propid = 1:9 (respectively, for label proportion = {0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9}), and rep = {1,2,3,4,5}; for each method and every label proportion, we repeat the experiments with 5 different initialization, and then take average. Due to time constraint, we had to separate them and run them in parallel. Also, sslda-trans is the code for transductive inference in sslda. (not covered in the paper)

In each folder, there is a python script ("PyRun.py") which takes care of all steps: training, test, and computing correct classification rate.

After running all experiments for each Dataset/Method, the python script "collect_results.py" in the Results folder takes average over all 5 iterations and save the final results in a .txt file.

