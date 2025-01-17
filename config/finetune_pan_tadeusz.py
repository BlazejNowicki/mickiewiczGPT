# finetune character-level model on Pan Tadeusz 
import time

out_dir = 'out-pan-tadeusz'
init_from = 'resume'
eval_interval = 100 
eval_iters = 200
log_interval = 10 

always_save_checkpoint = False

wandb_log = True 
wandb_project = 'pan-tadeusz-finetune'
wandb_run_name = 'mickiewiczGPT-' + str(time.time())

dataset = 'pan_tadeusz'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# the same parameters as during pretraining
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 2e-5 
max_iters = 65000
lr_decay_iters = 65000 # equal to max_iters 
min_lr = 1e-5 # learning_rate / 10 usually
beta2 = 0.95 

warmup_iters = 100 