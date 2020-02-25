import time

from importlib.machinery import SourceFileLoader

import wandb

from train_effnet import train_effnet

experiment = "tumor_detection"
#experiment = "MSI_classification"

cf = SourceFileLoader('cf', 'configs/config_tumor_detection.py').load_module()


# optimal parameter from wandb sweep for fine tuning
optimizing_parameters = cf.optimizing_parameters

timestr = time.strftime("%Y_%m_%d-%H:%M:%S")
debug = False
if debug:
    wandb.init(name=timestr, config=optimizing_parameters, project="first_aws")
else:
    wandb.init(name=timestr, config=optimizing_parameters, project=cf.project_name)

train_effnet(experiment=experiment, cf=cf, debug=debug)




