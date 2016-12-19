#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    * DESCRIPTION
    * REFERENCE
    * FEATURES
    * NEXT STEP

"""
########################################################################################################################
# IMPORT the necessary packages
import os
import sys
import json
import argparse


########################################################################################################################
#   SYSTEM PARAMETER

TRAIN_TEXT_FILE = "ms_marco.train.txt"

########################################################################################################################
#   CLASS


########################################################################################################################
#   LOCAL FUNCTIONS.

# ----------------------------------------------------------------------------------------------------------------------
def main(arg):

    if not os.path.exists(arg.dat_dir):
        print("\n @ Error: data directory no found, {}\n".format(arg.dat_dir))
        sys.exit()

    train_file = None
    for filename in os.listdir(arg.dat_dir):
        if filename.startswith("train_") and filename.endswith(".json"):
            train_file = filename
            break
    if not train_file:
        print("\n @ Error: training file not found.\n")
        sys.exit()
    train_ver = train_file.split('train_')[1].split(".json")[0]
    print(" % Training data version is {}".format(train_ver))

    json_data = []
    print(" # Read training json data...")
    with open(os.path.join(arg.dat_dir, train_file)) as fid:
        for line in fid:
            json_data.append(json.loads(line))
    pass

    cnt = 1
    fid = open(TRAIN_TEXT_FILE, 'w')
    print(" # Write training text data...")
    for line in json_data:
        for idx in range(len(line["answers"])):
            fid.write("{:<10d}  BB  100 {}\n".format(cnt, line["query"].replace('\n', ' ').encode('utf-8')))
            fid.write("{:<10d}  AA  200 {}\n".format(cnt, line["answers"][idx].replace('\n', ' ').encode('utf-8')))
            cnt += 1
    fid.close()

########################################################################################################################
########################################################################################################################
########################################################################################################################

if __name__ == "__main__":

    if len(sys.argv) == 1:
        sys.argv.extend(["--dir", "/opt/Dataset/MS.MARCO"])
        # sys.argv.extend(["--help"])
        pass

    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dir", dest="dat_dir", required=True, help="Directory having training data")

    args = parser.parse_args()

    main(args)
