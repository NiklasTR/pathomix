from importlib.machinery import SourceFileLoader
from multiprocessing import Manager, Process
import math
import numpy as np

import pandas as pd

from utils.utils import *

cf = SourceFileLoader('cf', 'configs/configs_mc.py').load_module()

costs_per_biomarker_checking = cf.costs_per_biomarker_checking
costs_per_pathomix_screening = cf.costs_per_pathomix_screening
profit_per_pathomix_screening = cf.profit_per_pathomix_screening

price_immun_ckpt_therapy = cf.price_immun_ckpt_therapy

ratio_of_patients_being_check = cf.ratio_of_patients_being_check

num_processes = cf.num_processes

load_file = cf.load_file
write_file = cf.write_file

df = pd.read_csv(load_file)

df = df.iloc[0:100000]

idxs = df.index

# split list if idx into equally long sublists
split_list = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]

sub_idx = split_list(idxs, num_processes)

# generate data that will represent our training data
manager = Manager()
results = manager.list()

# correct this
job = [Process(target=make_cost_calculations, args=(results, num_generators_per_process, auc_lower_bound, auc_upper_bound,
                                             num_training_patients, prevalences, sample_sizes, n_drawn_samples, sens_search,
                                             rdn_seed)) for rdn_seed in range(num_processes)]


_ = [p.start() for p in job]
_ = [p.join() for p in job]

result_df = results[0]

for df_part in results[1:]:
    result_df = result_df.append(df_part)

result_df.to_csv(write_file)