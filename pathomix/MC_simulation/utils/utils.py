from collections import OrderedDict
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as metr
from scipy.stats import poisson
import scipy as sp
import pandas as pd
import uuid


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


def make_cost_calculations(results, df, costs_per_biomarker_checking, costs_per_pathomix_screening,
                           profit_per_pathomix_screening,
                           price_immun_ckpt_therapy, ratio_of_patients_being_check):
    new_cols = ['baseline_revenue', 'ratio_of_patients_being_check', 'price_immun_ckpt_therapy',
                'profit_per_pathomix_screening',
                'profit_per_pathomix_screening', 'costs_per_biomarker_checking', 'pathomix_revenue_all',
                'pathomix_revenue_part',
                'patients_missed']
    # df_stage2 = pd.DataFrame([[np.NaN for _ in new_cols]], columns=new_cols)
    df_stage2 = pd.DataFrame()
    for cost_bm_c in costs_per_biomarker_checking:
        print('biomarker cost {}'.format(cost_bm_c))
        for profit_px_s in profit_per_pathomix_screening:
            for price_ickpt_t in price_immun_ckpt_therapy:
                for ratio in ratio_of_patients_being_check:
                    df_temp = pd.DataFrame(columns=new_cols)
                    # add base line costs
                    df_temp['baseline_revenue'] = (-df['sample_size'] * cost_bm_c + df[
                        'number_of_mutations'] * price_ickpt_t) / ratio
                    df_temp['ratio_of_patients_being_check'] = ratio
                    df_temp['price_immun_ckpt_therapy'] = price_ickpt_t
                    df_temp['profit_per_pathomix_screening'] = profit_px_s
                    df_temp['costs_per_biomarker_checking'] = cost_bm_c

                    # add pathomix revenue
                    df_temp['pathomix_revenue_all'] = -df['sample_size'] * costs_per_pathomix_screening - \
                                                      (df['sens'] * df['number_of_mutations'] + df['spez'] * (
                                                                  df['sample_size'] - df[
                                                              'number_of_mutations'])) * cost_bm_c + \
                                                      (df['sens'] * df['number_of_mutations'] * price_ickpt_t)
                    df_temp['pathomix_revenue_part'] = df_temp['pathomix_revenue_all'] / ratio

                    # add patients missed
                    df_temp['patients_missed'] = (1 - df['sens']) * df['number_of_mutations']
                    # print(df_temp.iloc[1])
                    df_temp['diff_baseline_px_part'] = df_temp['baseline_revenue'] - df_temp['pathomix_revenue_part']
                    df_temp['diff_baseline_px_all'] = df_temp['baseline_revenue'] - df_temp['pathomix_revenue_all']

                    if len(df_stage2) == 0:
                        df_stage2 = df_temp
                    else:
                        df_stage2 = df_stage2.append(df_temp, ignore_index=False)

    return results


def simulate_stage0(results, num_generate_train_distributions, auc_lower_bound, auc_upper_bound,
                             num_training_patients, prevalences, sample_sizes, n_drawn_samples, sens_search, rdn_seed):

    sp.random.seed()
    for n_generator in range(num_generate_train_distributions):
        uuid_generator = uuid.uuid4().hex
        #
        # training
        #
        print('n_generator {}'.format(n_generator))
        # assumption: we get training data from pharma. The distribution below is a theoretical gt for n = infinity
        # this step has to be performed to make some quality checks on the underlying distributions
        distributions_info = get_distributions_for_given_auc(auc_lower_bound, auc_upper_bound, num_training_patients,
                                                             num_training_patients, balanced=True)
        normal_theo = distributions_info['normal']
        mutations_theo = distributions_info['mutation']
        gts_theo = distributions_info['gts']
        theo_mu_normal = distributions_info['mu_normal']
        theo_mu_mutation = distributions_info['mu_mutation']
        theo_std_normal = distributions_info['std_normal']
        theo_std_mutation = distributions_info['std_mutation']

        fpr_theo, tpr_theo, thresholds_theo = get_roc_from_sample(normal_theo, mutations_theo,
                                                   gts_theo)  # TPR = sens, TNR = 1 - FPR = spez
        theo_auc = metr.auc(fpr_theo, tpr_theo)
        # plot_distributions(normal, mutations)

        #
        # sampling on "test" data
        #

        # now define thresholds for given sensitivity
        for prevalence in prevalences:
            for sas in sample_sizes:

                for dr in range(n_drawn_samples):
                    uuid_sample = uuid.uuid4().hex
                    #
                    #
                    #
                    '''
                    thres = {0.01: -2.1060045751689795,
                             0.05: -1.8042589799105322,
                             0.1: -1.6215225019891808}
                    '''
                    # draw sample with sas patiente from theoretical distribution
                    # number_normal_cases = sas - number_mutations
                    number_mutations = poisson.rvs(mu=sas * prevalence)
                    if number_mutations==0: # is no muatoins are available fill dict and do not sample but continue with loop
                        result = OrderedDict()
                        result['uuid_gen'] = uuid_generator
                        result['uuid_sample'] = uuid_sample
                        result['theo_auc'] = theo_auc
                        result['train_auc'] = np.NaN
                        result['sample_size'] = sas
                        result['prevalence'] = prevalence
                        result['mu_normal'] = theo_mu_normal
                        result['mu_mutation'] = theo_mu_mutation
                        result['std_normal'] = theo_std_normal
                        result['std_mutation'] = theo_std_mutation
                        result['number_of_mutations'] = number_mutations
                        result['threshold_training'] = np.NaN
                        #result['tn'] = np.NaN
                        #result['fp'] = np.NaN
                        #result['fn'] = np.NaN
                        #result['tp'] = np.NaN
                        result['sens'] = np.NaN
                        result['spez'] = np.NaN
                        result['ppv'] = np.NaN
                        result['npv'] = np.NaN
                        result['fnr'] = np.NaN
                    else:

                        sample_summary = sample_data(sas, number_mutations, mu_mutation=theo_mu_mutation,
                                                     mu_normal=theo_mu_normal, std_mutation=theo_std_mutation,
                                                     std_normal=theo_std_normal, balanced=False)
                        normal = sample_summary['normal']
                        mutation = sample_summary['mutation']
                        gts = sample_summary['gts']

                        # now define thresholds for given sensitivity
                        fpr, tpr, thresholds = get_roc_from_sample(normal, mutation,
                                                                   gts)  # TPR = sens, TNR = 1 - FPR = spez
                        train_auc = metr.auc(fpr, tpr)
                        thres = {}

                        for s in sens_search:
                            #idx = get_index_for_threshold_sens(tpr, s)
                            idx = get_index_for_threshold_spec(fpr, s)
                            thres[s] = thresholds[idx]  # loop over this dict in later monte carlo

                        for k, t in thres.items():
                            # print('sample size {}, draw {}, threshold {}'.format(sas, dr, threshold))
                            result = OrderedDict()
                            result['uuid_gen'] = uuid_generator
                            result['uuid_sample'] = uuid_sample
                            result['theo_auc'] = theo_auc
                            result['train_auc'] = train_auc
                            result['sample_size'] = sas
                            result['prevalence'] = prevalence
                            result['mu_normal'] = theo_mu_normal
                            result['mu_mutation'] = theo_mu_mutation
                            result['std_normal'] = theo_std_normal
                            result['std_mutation'] = theo_std_mutation

                            result['number_of_mutations'] = number_mutations

                            result['spec_training'] = k
                            # plot_distributions(normal, mutation)
                            # here we apply the thresholds determined using only n patients. This threshold will now be
                            # applied to the theoretical distribution and we will get how many FN, TP, ... patients
                            # we would get in a testing scenario
                            summary_train = summarize_preds(normal_theo, mutations_theo, t)
                            for k, v in summary_train.items():
                                result[k] = v
                            results.append(result)
    return results


def get_index_for_threshold_sens(tpr, sens):
    # tpr == sens  = 1- FNR (https://duckduckgo.com/?q=sensitivity&t=canonical)
    diff = abs(tpr - sens)
    diff_reverse = diff[::-1]  # since fpr, tpr and threshold are right to left
    return abs(np.argmin(diff_reverse) - len(tpr) + 1)  # subtract length of array to be in right order again


def get_index_for_threshold_spec(fpr, spec):
    spec_is = 1- fpr
    # tpr == sens  = 1- FNR (https://duckduckgo.com/?q=sensitivity&t=canonical)
    diff = abs(spec_is - spec)
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
        ppv = np.NaN
    if not (tn==0 and fn==0):
        npv = tn / (tn + fn)
    else:
        npv = np.NaN
    fnr = fn / (tp + fn)

    summary = OrderedDict()
    #summary['tn'] = tn
    #summary['fp'] = fp
    #summary['fn'] = fn
    #summary['tp'] = tp
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