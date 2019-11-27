# define parameters to get "training" distributions
# important assumptions: 1) training and test data will be from same distribution
num_training_patients = 10000 # how many patients will be in training data set, choose large to have a smooth distribution
# pick realistic range for AUC -> many random distributions will be generated. If the AUC for a distribution is
# auc_lower_bound < auc < auc_upper_bound, the distribution will be picked ro further analysis
auc_lower_bound = 0.6
auc_upper_bound = 0.9
num_generate_train_distributions = 2 # how many training/true distributions will be created
# sens_search: get a threshold on the training distribution for the given sensitivity.
# this threshold will then be applied to the testing distribution
sens_search = [0.995, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]

# now prepare configs for testing (draw randomly from the training/true distibution)
prevalence = prevalences = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5] # biomarker specific
sample_sizes = [50, 100, 200, 500, 1000]

wt_training_pats = num_training_patients * (1-prevalence)
mut_training_pats = num_training_patients * prevalence
fpr = 0.01 # fix threshold for classifiction on training distribution. This threshold will then be used at test time

n_drawn_samples = 100 # how often samples will be drawn for a given training distribution


# parameters at test time
pat_per_year_at_cc = 10000
pat_per_month = pat_per_year_at_cc/12.

avg_trial_size = 100
cost_squencing = 500  # dollar
cost_evaluation = 30 # dollar
cost_identification = 10 # dollar ?
cost_pathomix_prescreening = 5 # dollar

drug_profit_per_year = 3500000000 # dollar

identification_rate = 0.1
evaluation_rate = 0.1