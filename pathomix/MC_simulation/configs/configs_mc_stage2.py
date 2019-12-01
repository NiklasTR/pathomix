# all prices in dollar
costs_per_biomarker_checking = [50, 100, 200, 500]
costs_per_pathomix_screening = 5
profit_per_pathomix_screening = [1, 20, 100, 1000]

price_immun_ckpt_therapy = [10*5,  # imaginary price
                           1.1 * 10**6,  # for base line nivolumab check https://onlinelibrary.wiley.com/doi/full/10.1002/cncr.31795 table2
                           3 * 10**6]  # for nivolumab and Ipilimumab

ratio_of_patients_being_check = [0.25, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99]

load_file ='/home/ubuntu/bucket/pathomix/results/simulations/stage0_v2.csv'

write_file = '/home/ubuntu/bucket/pathomix/results/simulations/stage2.csv'