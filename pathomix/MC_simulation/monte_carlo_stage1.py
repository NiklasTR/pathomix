from importlib.machinery import SourceFileLoader
from multiprocessing import Manager, Process
import math

import pandas as pd

from pathomix.MC_simulation.utils.utils import simulate_stage0

cf = SourceFileLoader('cf', 'configs/configs_mc.py').load_module()

num_generate_train_distributions = cf.num_generate_train_distributions
auc_lower_bound = cf.auc_lower_bound
auc_upper_bound = cf.auc_upper_bound
num_training_patients = cf.num_training_patients
n_drawn_samples = cf.n_drawn_samples
prevalences = cf.prevalences
sample_sizes = cf.sample_sizes
#sens_search = cf.sens_search
sens_search = cf.spec_search

num_processes = cf.num_processes

num_generators_per_process = math.ceil(num_generate_train_distributions/num_processes)

out_file = cf.out_file



# generate data that will represent our training data
manager = Manager()
results = manager.list()

job = [Process(target=simulate_stage0, args=(results, num_generators_per_process, auc_lower_bound, auc_upper_bound,
                                             num_training_patients, prevalences, sample_sizes, n_drawn_samples, sens_search,
                                             rdn_seed)) for rdn_seed in range(num_processes)]


_ = [p.start() for p in job]
_ = [p.join() for p in job]

# create pandas data frame holding all results
results_list = list(results)
df = pd.DataFrame(results_list)
df.to_csv(out_file)