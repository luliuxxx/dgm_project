#python training.py --data_flag pathmnist --use_config CVAE_RGB --label_to_binary "cancer-associated stroma" --max_epochs 100 --wandb 1   
#python training.py --data_flag dermamnist --use_config CVAE_RGB --label_to_binary "basal cell carcinoma" --max_epochs 100 --wandb 1
#python training.py --data_flag bloodmnist --use_config CVAE_RGB --label_to_binary "erythroblast" --max_epochs 100 --wandb 1


python evaluation.py --data_flag pathmnist --use_config CVAE_RGB --label_to_binary "cancer-associated stroma" --ckpt_name cvae_pathmnist_final.pt
python evaluation.py --data_flag dermamnist --use_config CVAE_RGB --label_to_binary "basal cell carcinoma" --ckpt_name cvae_dermamnist_final.pt
python evaluation.py --data_flag bloodmnist --use_config CVAE_RGB --label_to_binary "erythroblast" --ckpt_name cvae_bloodmnist_final.pt

