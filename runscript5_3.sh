
#!/usr/bin/bash

declare -a datasets=("ZINC")
for dataset in ${datasets[@]}
do
# normal, beta=0.1
#python manage.py DEL --dataset ${dataset} --lr 0.0001 --step_size 4 --gamma 0.8 --batch_size 128 --embed_size 128 --hidden_size 128 --hidden_layer 2 --latent_size 64 --random_seed 10000 --use_gpu --validate_after 0.0001 --validation_samples 10 --predictor_num_layers 2 --predictor_hidden_size 64 --k_beta 1 --a_beta 0.1 --l_beta 0.1 --u_beta 1 --increase_beta 1 --k_alpha 1 --a_alpha 1 --l_alpha 1 --u_alpha 100 --increase_alpha 1 --num_generations 10 --population_size 30000 --init_num_epochs 10 --subsequent_num_epochs 5 --save_pops --prob_ts 0.95 --crossover linear --mutation 0.01

# beta = 0.01
# python manage.py DEL --dataset ${dataset} --lr 0.0001 --step_size 4 --gamma 0.8 --batch_size 128 --embed_size 128 --hidden_size 128 --hidden_layer 2 --latent_size 64 --random_seed 10000 --use_gpu --validate_after 0.0001 --validation_samples 10 --predictor_num_layers 2 --predictor_hidden_size 64 --k_beta 1 --a_beta 0.01 --l_beta 0.01 --u_beta 1 --increase_beta 1 --k_alpha 1 --a_alpha 1 --l_alpha 1 --u_alpha 100 --increase_alpha 1 --num_generations 10 --population_size 30000 --init_num_epochs 20 --subsequent_num_epochs 5 --save_pops --prob_ts 0.95 --crossover linear --mutation 0.01

# test beta for beta = 0.01
# python manage.py DEL --dataset ${dataset} --lr 0.001 --step_size 4 --gamma 0.8 --batch_size 32 --embed_size 128 --hidden_size 450 --hidden_layer 1 --latent_size 64 --random_seed 10000 --use_gpu --validate_after 0.0001 --validation_samples 10 --predictor_num_layers 2 --predictor_hidden_size 64 --k_beta 1 --a_beta 0.01 --l_beta 0.01 --u_beta 1 --increase_beta 1 --k_alpha 1 --a_alpha 1 --l_alpha 1 --u_alpha 100 --increase_alpha 1 --num_generations 10 --population_size 20000 --init_num_epochs 10 --subsequent_num_epochs 5 --save_pops --prob_ts 0.95 --crossover linear --mutation 0.01

# run for beta = 0.1,  population = 20k
python manage.py DEL --dataset ${dataset} --lr 0.0001 --step_size 4 --gamma 0.8 --batch_size 32 --embed_size 128 --hidden_size 450 --hidden_layer 1 --latent_size 64 --random_seed 10000 --use_gpu --validate_after 0.0001 --validation_samples 10 --predictor_num_layers 2 --predictor_hidden_size 64 --k_beta 1 --a_beta 0.1 --l_beta 0.1 --u_beta 1 --increase_beta 1 --k_alpha 1 --a_alpha 1 --l_alpha 1 --u_alpha 100 --increase_alpha 1 --num_generations 10 --population_size 20000 --init_num_epochs 20 --subsequent_num_epochs 5 --save_pops --prob_ts 0.95 --crossover linear --mutation 0.01

# test for SOPSO
# python manage_sopso.py DEL --dataset ${dataset} --lr 0.0001 --step_size 4 --gamma 0.8 --batch_size 32 --embed_size 128 --hidden_size 450 --hidden_layer 1 --latent_size 64 --random_seed 10000 --use_gpu --validate_after 0.0001 --validation_samples 10 --predictor_num_layers 2 --predictor_hidden_size 64 --k_beta 1 --a_beta 0.1 --l_beta 0.01 --u_beta 1 --increase_beta 1 --k_alpha 1 --a_alpha 1 --l_alpha 1 --u_alpha 100 --increase_alpha 1 --num_generations 10 --population_size 20000 --init_num_epochs 10 --subsequent_num_epochs 5 --save_pops --prob_ts 0.95 --crossover linear --mutation 0.01 --single_objective

# beta from 0.01 to 0.4, alpha from 1 to 4
#python manage.py DEL --dataset ${dataset} --lr 0.0001 --step_size 4 --gamma 0.8 --batch_size 128 --embed_size 128 --hidden_size 128 --hidden_layer 2 --latent_size 64 --random_seed 10000 --use_gpu --validate_after 0.0001 --validation_samples 10 --predictor_num_layers 2 --predictor_hidden_size 64 --k_beta 1 --a_beta 0.01 --l_beta 0.01 --u_beta 1 --increase_beta 40 --k_alpha 1 --a_alpha 1 --l_alpha 1 --u_alpha 100 --increase_alpha 4 --num_generations 10 --population_size 30000 --init_num_epochs 10 --subsequent_num_epochs 5 --save_pops --prob_ts 0.95 --crossover linear --mutation 0.01

# beta from 0.1 to 0.4, alpha from 1 to 4
#python manage.py DEL --dataset ${dataset} --lr 0.0001 --step_size 4 --gamma 0.8 --batch_size 128 --embed_size 128 --hidden_size 128 --hidden_layer 2 --latent_size 64 --random_seed 10000 --use_gpu --validate_after 0.0001 --validation_samples 10 --predictor_num_layers 2 --predictor_hidden_size 64 --k_beta 1 --a_beta 0.1 --l_beta 0.1 --u_beta 1 --increase_beta 4 --k_alpha 1 --a_alpha 1 --l_alpha 1 --u_alpha 100 --increase_alpha 4 --num_generations 10 --population_size 20000 --init_num_epochs 10 --subsequent_num_epochs 5 --save_pops --prob_ts 0.95 --crossover linear --mutation 0.01

# 100K samples
#python manage.py DEL --dataset ${dataset} --lr 0.0001 --step_size 4 --gamma 0.8 --batch_size 128 --embed_size 128 --hidden_size 128 --hidden_layer 2 --latent_size 64 --random_seed 10000 --use_gpu --validate_after 0.0001 --validation_samples 10 --predictor_num_layers 2 --predictor_hidden_size 64 --k_beta 1 --a_beta 0.1 --l_beta 0.1 --u_beta 1 --increase_beta 4 --k_alpha 1 --a_alpha 1 --l_alpha 1 --u_alpha 100 --increase_alpha 4 --num_generations 10 --population_size 100000 --init_num_epochs 50 --subsequent_num_epochs 25 --save_pops --prob_ts 0.95 --crossover linear --mutation 0.01

done

#for d in ./RUNS/*
#do
#    python manage.py plot_del --run $d
#done






