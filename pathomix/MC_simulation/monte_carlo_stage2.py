from importlib.machinery import SourceFileLoader
from multiprocessing import Manager, Process
import math
import numpy as np

import pandas as pd

from utils.utils import *

cf = SourceFileLoader('cf', 'configs/configs_mc_stage2.py').load_module()

costs_per_biomarker_checking = cf.costs_per_biomarker_checking
costs_per_pathomix_screening = cf.costs_per_pathomix_screening
profit_per_pathomix_screening = cf.profit_per_pathomix_screening

price_immun_ckpt_therapy = cf.price_immun_ckpt_therapy

ratio_of_patients_being_check = cf.ratio_of_patients_being_check

load_file = cf.load_file
write_file = cf.write_file

df = pd.read_csv(load_file)

new_cols = ['baseline_revenue', 'ratio_of_patients_being_check', 'price_immun_ckpt_therapy', 'profit_per_pathomix_screening',
                                  'profit_per_pathomix_screening', 'costs_per_biomarker_checking', 'pathomix_revenue_all', 'pathomix_revenue_part',
                                  'patients_missed']
#df_stage2 = pd.DataFrame([[np.NaN for _ in new_cols]], columns=new_cols)
df_stage2 = pd.DataFrame()
for cost_bm_c in costs_per_biomarker_checking:
    print('biomarker cost {}'.format(cost_bm_c))
    for profit_px_s in profit_per_pathomix_screening:
        for price_ickpt_t in price_immun_ckpt_therapy:
            for ratio in ratio_of_patients_being_check:
                df_temp = pd.DataFrame(columns=new_cols)
                # add base line costs
                df_temp['baseline_revenue'] = (-df['sample_size'] * cost_bm_c + df['number_of_mutations'] * price_ickpt_t) / ratio
                df_temp['ratio_of_patients_being_check'] = ratio
                df_temp['price_immun_ckpt_therapy'] = price_ickpt_t
                df_temp['profit_per_pathomix_screening'] = profit_px_s
                df_temp['costs_per_biomarker_checking'] = cost_bm_c

                # add pathomix revenue
                df_temp['pathomix_revenue_all'] = -df['sample_size'] * costs_per_pathomix_screening - \
                                                     (df['sens']* df['number_of_mutations'] + df['spez'] * (df['sample_size'] - df['number_of_mutations'])) * cost_bm_c + \
                                                     (df['sens'] * df['number_of_mutations'] * price_ickpt_t)
                df_temp['pathomix_revenue_part'] = df_temp['pathomix_revenue_all'] / ratio

                # add patients missed
                df_temp['patients_missed'] = (1 - df['sens']) * df['number_of_mutations']
                #print(df_temp.iloc[1])
                df_temp['diff_baseline_px_part'] = df_temp['baseline_revenue'] - df_temp['pathomix_revenue_part']
                df_temp['diff_baseline_px_all'] = df_temp['baseline_revenue'] - df_temp['pathomix_revenue_all']

                if len (df_stage2) == 0:
                    df_stage2 = df_temp
                else:
                    df_stage2 = df_stage2.append(df_temp, ignore_index=False)

df_stage2.to_csv(write_file)