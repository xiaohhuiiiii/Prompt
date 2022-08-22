train_param = {
    'loss': 'CE', 
    'batch_size': 128,
    'lr': 0.1,
    'weight_decay': 1e-4,
    'epoch': 1000,
    'print_freq': 10, 
    'model': 'prompt', 
    'run_name': 'prompt', 
    'loss_weight': None, 
    'prompt': 'prompt', 
    'augmentation': ['randomcrop', 'randomflip'], 
    'backbond': 'rn101'
}

val_param = {
    'batch_size': 64,
    'print_freq': 11,
}

use_wandb = True
if_save = True
project_name = 'aug_for_prompt'