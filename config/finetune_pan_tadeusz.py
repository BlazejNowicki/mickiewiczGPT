# finetune character-level model on Pan Tadeusz 

out_dir = 'out-pan-tadeusz'
init_from = 'resume'
eval_interval = 250 
eval_iters = 200
log_interval = 10 

# 
always_save_checkpoint = False

wandb_log = True 
wandb_project = 'pan-tadeusz-finetune'
wandb_run_name = 'mini-gpt'

dataset = 'pan_tadeusz_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# the same parameters as during pretraining
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 5e-5 
max_iters = 10000
lr_decay_iters = 10000 # equal to max_iters 
min_lr = 1e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 