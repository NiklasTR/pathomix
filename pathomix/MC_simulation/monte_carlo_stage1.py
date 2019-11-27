from collections import OrderedDict
from importlib.machinery import SourceFileLoader

import pandas as pd
import sklearn.metrics as metr
from scipy.stats import poisson
from utils.utils import *

cf = SourceFileLoader('cf', 'configs/configs_mc_stage0.py').load_module()

num_generate_train_distributions = cf.num_generate_train_distributions
auc_lower_bound = cf.auc_lower_bound
auc_upper_bound = cf.auc_upper_bound
num_training_patients = cf.num_training_patients
n_drawn_samples = cf.n_drawn_samples



# generate data that will represent our training data
results = []
for n_generator in range(num_generate_train_distributions):
    #
    # training
    #
    print('n_generator {}'.format(n_generator))
    distributions_info = get_distributions_for_given_auc(auc_lower_bound, auc_upper_bound, num_training_patients,
                                                         num_training_patients, balanced=True)
    normal = distributions_info['normal']
    mutations = distributions_info['mutation']
    gts = distributions_info['gts']
    train_mu_normal = distributions_info['mu_normal']
    train_mu_mutation = distributions_info['mu_mutation']
    train_std_normal = distributions_info['std_normal']
    train_std_mutation = distributions_info['std_mutation']
    # plot_distributions(normal, mutations)

    # now define thresholds for given sensitivity
    fpr, tpr, thresholds = get_roc_from_sample(normal, mutations, gts)
    train_auc = metr.auc(fpr, tpr)
    thres = {}
    sens_search = [0.995, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5,
                   0.4, 0.3, 0.2, 0.1, 0.05, 0.01]

    for t in sens_search:
        idx = get_index_for_threshold(fpr, t)
        thres[t] = thresholds[idx]  # loop over this dict in later monte carlo

    #
    # sampling on "test" data
    #

    # now define thresholds for given sensitivity
    prevalences = [0.01, 0.03, 0.05, 0.1, 0.2]

    for prevalence in prevalences:
        sample_sizes = [50, 100, 200, 500, 1000]

        for sas in sample_sizes:
            for dr in range(n_drawn_samples):
                #
                #
                #
                '''
                thres = {0.01: -2.1060045751689795,
                         0.05: -1.8042589799105322,
                         0.1: -1.6215225019891808}
                '''
                for threshold in thres:
                    # print('sample size {}, draw {}, threshold {}'.format(sas, dr, threshold))
                    result = OrderedDict()
                    result['auc_training'] = train_auc
                    result['threshold_training'] = threshold
                    result['sample_size'] = sas
                    result['prevalence'] = prevalence
                    result['mu_normal'] = train_mu_normal
                    result['mu_mutation'] = train_mu_mutation
                    result['std_normal'] = train_std_normal
                    result['std_mutation'] = train_std_mutation
                    number_mutations = poisson.rvs(mu=sas * prevalence)
                    result['number_of_mutations'] = number_mutations
                    # print("{} of {} patients have a mutation".format(number_mutations, sas))
                    if number_mutations == 0:
                        result['auc'] = 0
                        result['tn'] = 0
                        result['fp'] = 0
                        result['fn'] = 0
                        result['tp'] = 0
                        result['sens'] = 0
                        result['spez'] = 0
                        result['ppv'] = 0
                        result['npv'] = 0
                        result['fnr'] = 0
                    else:
                        number_normal_cases = sas - number_mutations
                        sample_summary = sample_data(sas, number_mutations, mu_mutation=train_mu_mutation,
                                                     mu_normal=train_mu_normal, std_mutation=train_std_mutation,
                                                     std_normal=train_std_normal, balanced=False)
                        normal = sample_summary['normal']
                        mutation = sample_summary['mutation']
                        gts = sample_summary['gts']

                        fpr, tpr, thresholds = get_roc_from_sample(normal, mutation, gts)
                        result['auc'] = get_auc(normal, mutation, gts, False)
                        # plot_distributions(normal, mutation)
                        summary_train = summarize_preds(normal, mutation, threshold)
                        for k, v in summary_train.items():
                            result[k] = v
                    results.append(result)

# create pandas data frame holding all results
df = pd.DataFrame(results)
df.to_csv('.results/stage0.csv')