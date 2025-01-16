# train a character-level model on polish literature

out_dir = 'out-pan-tadeusz'
eval_interval = 500 
eval_iters = 100
log_interval = 10

always_save_checkpoint = True

wandb_log = True 
wandb_project = 'pan-tadeusz-pretrain'
wandb_run_name = 'mini-gpt'

dataset = 'polish_literature'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-4
max_iters = 100000
lr_decay_iters = 100000 # make equal to max_iters usually
min_lr = 1e-5 # learning_rate / 10 usually
beta2 = 0.97 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 
