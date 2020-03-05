import pandas as pd
import numpy as np
import subprocess

from pathomix.preprocessing.tools import slide_utils

'''
file_path = '/home/pmf/Documents/DataMining/pathomix/data/TMB_tables/explore-case-table_2020-02-26_total.csv'
df = pd.read_table(file_path)
df['TMB_corrected'] = df['# Mutations']/38. # exon is estimated to be 38MB big
df['TMB_corrected_log'] = np.log(df['TMB_corrected'])
'''

if __name__ == '__main__':
    file_path = '/home/pmf/Documents/DataMining/pathomix/data/TMB_tables/explore-case-table_2020-02-26_total.csv'
    df = pd.read_table(file_path)
    df['TMB_corrected'] = df['# Mutations'] / 38.  # exon is estimated to be 38MB big
    df['TMB_corrected_log'] = np.log(df['TMB_corrected'])

    # loop over patient for which we have TMB data

    for case_id in df['Case ID'][:1]:
        subprocess.run(["aws", "s3", "sync", "s3://evotec/pathomix/data/TCGA/COAD/*/{}*".format(case_id),
                        "/home/ubuntu/bucket/TMB_preparation/{}".format(case_id)])

        #tile_summaries =
