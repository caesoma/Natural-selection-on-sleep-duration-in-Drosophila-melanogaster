#!/bin/bash

# the "wrangle.py" script takes the initial and final generations as the first two arguments (12 and 75 in this example), splits a table based on those generations (last argument, here "Allchrs_All_lines_MAF_G12_G75.csv"), joins them by matching the metadata, as randomly selects a subset of the sites based on the third argument ("cmh_significant_sites.csv" in this example)

python3 wrangle.py 12 75 cmh_significant_sites.csv all_sites.csv
# in this script it outputs the subset table "subset_all_sites_innersplit.csv", used below

# "main.py" takes the effective population size and number of generations as the first two arguments (43 and 63 here, respectively), computes the drift probabilities (if not previously computed and with name expected by the script in `data_dir` folder) for initial and final frequencies in columns named with arguments 3 and 4, computes likelihood for each site, and saving tables in arguments 5 and 6 with that column appended

python3 main.py 42 63 MAF12 MAF75 cmh_significant_sites.csv subset_all_sites_innersplit.csv

# plots likelihoods by selection and replicate combinations from results computed by revious scripts
python3 plot.py cmh_significant_sites_likelihoods.csv subset_all_sites_innersplit_likelihoods.csv MAF75-MAF12

# * allele frequency column may have any name but must be consistent between variants of interest and rest of polymorphism, which must be provided to `main.py` (in the example original column is MAF in unsplit table, which becomes MAF12 and MAF75 after splitting)
