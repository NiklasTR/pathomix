from collections import OrderedDict
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as metr
from scipy.stats import poisson
import scipy as sp

'''
def calculate_pathomix_stage2_per_month(prevalence, pat_per_month, mu_mutation, mu_normal, std_mutation, std_normal,
                                        balanced=False):
    expected_number_of_mutations = prevalence * pat_per_month
    number_mutations = poisson.rvs(
        mu=expected_number_of_mutations)  # draw actual number of patients with mutation from poisson distr

    distribution_info_test = sample_data(pat_per_month, number_mutations, mu_mutation=mu_mutation, mu_normal=mu_normal,
                                         std_mutation=std_mutation, std_normal=std_normal, balanced=balanced)
    normal_test = distribution_info_test['normal']
    mutation_test = distribution_info_test['mutation']
    plot_distributions(normal_test, mutation_test, threshold)
    summary_testing = summarize_preds(normal_test, mutation_test, threshold)
    print('missed patients with mutation {}'.format(summary_testing['fn']))
    # choose patients that are positively predicted
    normal_test = get_pos_pred_patients(normal_test, threshold)
    mutation_test = get_pos_pred_patients(mutation_test, threshold)

    cost_total, costs_sequencing, num_neg_pat, num_pos_pat = pathomix_stage2_cost_workflow(pat_per_month,
                                                                                           cost_pathomix_prescreening,
                                                                                           cost_identification,
                                                                                           identification_rate,
                                                                                           cost_evaluation,
                                                                                           evaluation_rate, normal_test,
                                                                                           mutation_test)
    return cost_total, costs_sequencing, num_neg_pat, num_pos_pat
'''



def simulate_stage0(results, num_generate_train_distributions, auc_lower_bound, auc_upper_bound,
                             num_training_patients, prevalences, sample_sizes, n_drawn_samples, sens_search, rdn_seed):

    sp.random.seed()
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

        for t in sens_search:
            idx = get_index_for_threshold(fpr, t)
            thres[t] = thresholds[idx]  # loop over this dict in later monte carlo

        #
        # sampling on "test" data
        #

        # now define thresholds for given sensitivity
        for prevalence in prevalences:
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
                            #number_normal_cases = sas - number_mutations
                            sample_summary = sample_data(sas, number_mutations, mu_mutation=train_mu_mutation,
                                                         mu_normal=train_mu_normal, std_mutation=train_std_mutation,
                                                         std_normal=train_std_normal, balanced=False)
                            normal = sample_summary['normal']
                            mutation = sample_summary['mutation']
                            gts = sample_summary['gts']

                            #fpr, tpr, thresholds = get_roc_from_sample(normal, mutation, gts)
                            result['auc'] = get_auc(normal, mutation, gts, False)
                            # plot_distributions(normal, mutation)
                            summary_train = summarize_preds(normal, mutation, threshold)
                            for k, v in summary_train.items():
                                result[k] = v
                        results.append(result)
    return results


def get_index_for_threshold(fpr, sens):
    t = 1 - fpr  # == sens
    diff = abs(t - sens)
    diff_reverse = diff[::-1]  # since fpr, tpr and threshold are right to left
    return abs(np.argmin(diff_reverse) - len(fpr) + 1)  # subtract length of array to be in right order again

'''
def pathomix_stage2_cost_workflow(pat_per_month, cost_pathomix_prescreening, cost_identification, identification_rate,
                                  cost_evaluation, evaluation_rate, normal_test, mutation_test):
    pmx_costs_prescreening = pat_per_month * cost_pathomix_prescreening
    pmx_costs_identifing = (len(normal_test) + len(mutation_test)) * cost_identification

    pmx_identifed_normal = patients_identified(normal_test, identification_rate)
    pmx_identifed_mutation = patients_identified(mutation_test, identifiation_rate)

    pmx_costs_eval = (len(pmx_identifed_normal) + len(pmx_identifed_mutation)) * cost_evaluation

    pmx_evaluated_normal = patients_evaluated(pmx_identifed_normal, evaluation_rate)
    pmx_evaluated_mutated = patients_evaluated(pmx_identifed_mutation, evaluation_rate)

    pmx_sequencing_costs = (len(pmx_evaluated_normal) + len(pmx_evaluated_mutated)) * cost_squencing

    pmx_costs = pmx_costs_prescreening + pmx_costs_identifing + pmx_costs_eval + pmx_sequencing_costs

    print('sequencing costs for one month: {}'.format(pmx_sequencing_costs))
    print('total costs for one month: {}'.format(pmx_costs))

    return pmx_costs, pmx_sequencing_costs, len(pmx_evaluated_normal), len(pmx_evaluated_mutated)
'''

def get_pos_pred_patients(patients, threshold):
    return patients[patients > threshold]


def patients_identified(patients, identify_rate=0.1):
    return np.random.choice(patients, replace=True, size=int(len(patients) * identify_rate))


def patients_evaluated(patients, eval_rate=0.1):
    return np.random.choice(patients, replace=True, size=int(len(patients) * eval_rate))


def get_distributions_for_given_auc(lower_bound, upper_bound, n_per_month, number_mutations, balanced=False,
                                    plot_AUC=False):
    auc = -1
    while not lower_bound < auc < upper_bound:
        summary = sample_data(n_per_month, number_mutations, balanced=balanced)
        normal = summary['normal']
        mutations = summary['mutation']
        gts = summary['gts']

        auc = get_auc(normal, mutations, gts, plot_AUC)
    return summary


def get_pred(arr, thresh):
    pred = np.zeros(len(arr))
    pred[arr > thresh] = 1
    return pred


def summarize_preds(arr_neg, arr_pos, threshold, b_print=False):
    pred_a = get_pred(arr_neg, threshold)
    pred_b = get_pred(arr_pos, threshold)
    gt_a = np.zeros(len(arr_neg))
    gt_b = np.ones(len(arr_pos))

    preds = np.append(pred_a, pred_b)
    gts = np.append(gt_a, gt_b)

    tn, fp, fn, tp = metr.confusion_matrix(gts, preds).ravel()
    sens = tp / (tp + fn)
    spez = tn / (tn + fp)
    if not (tp==0 and fp ==0):
        ppv = tp / (tp + fp)
    else:
        ppv = 0
    if not (tn==0 and fn==0):
        npv = tn / (tn + fn)
    else:
        npv = 0
    fnr = fn / (tp + fn)

    summary = OrderedDict()
    summary['tn'] = tn
    summary['fp'] = fp
    summary['fn'] = fn
    summary['tp'] = tp
    summary['sens'] = sens
    summary['spez'] = spez
    summary['ppv'] = ppv
    summary['npv'] = npv
    summary['fnr'] = fnr

    if b_print:
        print('tn: {}'.format(tn))
        print('fp: {}'.format(fp))
        print('fn: {}'.format(fn))
        print('tp: {}'.format(tp))
        print('sens: {}'.format(sens))
        print('spez: {}'.format(spez))
        print('ppv: {}'.format(ppv))
        print('npv: {}'.format(npv))
        print('fnr: {}'.format(fnr))

    return summary


def get_threshold_for_fnr(arr_pos, fnr=0.01):
    histo, thresholds = np.histogram(arr_pos, bins=len(arr_pos))
    fnr_numerical = 0
    idx = -1
    while fnr_numerical < fnr:
        idx += 1
        fnr_numerical += histo[idx] / np.sum(histo)
    print('wanted FNR: {}, got {}'.format(fnr, fnr_numerical))

    return thresholds[idx + 1]


def get_roc_from_sample(normal, mutation, gts):
    # clf = LinearDiscriminantAnalysis()
    # clf.fit(np.append(normal,mutation)[:,None], gts)
    # preds = clf.predict(np.append(a,b)[:,None])
    # preds_proba = clf.predict_proba(np.append(normal ,mutation)[:,None])

    fpr, tpr, thresholds = metr.roc_curve(gts, np.append(normal, mutation), pos_label=1, drop_intermediate=False)

    return fpr, tpr, thresholds


def get_auc(normal, mutation, gts, plot_auc=True):
    fpr, tpr, _ = get_roc_from_sample(normal, mutation, gts)
    auc = metr.auc(fpr, tpr)
    if plot_auc:
        print('AUC: {}'.format(auc))
        plot_roc_cur(fpr, tpr)

    return auc


def sample_data(n_per_month, number_mutations, mu_mutation=None, mu_normal=None, std_mutation=None, std_normal=None,
                balanced=False):
    if not mu_normal:
        mu_mut = -1
        mu_normal = 0
        # make sure class 1 (mutation) is to the right of class 0
        while mu_normal > mu_mut:
            mu_mut, mu_normal, std_mut, std_normal = np.random.normal(loc=0, scale=1, size=4)
            std_mut = abs(std_mut)
            std_normal = abs(std_normal)
    else:
        mu_mut, mu_normal, std_mut, std_normal = mu_mutation, mu_normal, std_mutation, std_normal

    if not balanced:
        normal = np.random.normal(mu_normal, abs(std_normal), int(n_per_month - number_mutations))
        mutation = np.random.normal(mu_mut, abs(std_mut), int(number_mutations))
    else:
        normal = np.random.normal(mu_normal, abs(std_normal), int(n_per_month))
        mutation = np.random.normal(mu_mut, abs(std_mut), int(n_per_month))

    gt_normal = np.zeros_like(normal)
    gt_mutation = np.ones_like(mutation)
    gts = np.append(gt_normal, gt_mutation)

    summary = {}
    summary['normal'] = normal
    summary['mutation'] = mutation
    summary['gts'] = gts
    summary['mu_mutation'] = mu_mut
    summary['mu_normal'] = mu_normal
    summary['std_mutation'] = std_mut
    summary['std_normal'] = std_normal

    return summary


def plot_roc_cur(fper, tper):
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


def plot_distributions(normal, mutation, threshold=None):
    sns.distplot(normal, kde=False)
    sns.distplot(mutation, kde=False)
    if threshold:
        plt.axvline(threshold)
    plt.show()
    # sns.distplot(np.append(normal,b), kde=False)