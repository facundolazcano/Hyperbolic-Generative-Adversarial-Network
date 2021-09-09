#!/bin/bash
#eeee_eeem_tuning_3_seed_500
cfg="eeeeeethh"
c="0.0001"
seed_train="1"
ckpt_path="/home/jenny2/HypStyleGAN/Nuevos_exp/eeee_eethh_tuning_3_c1e-4/checkpoint/200000.pt"
device=2
path_sample="/home/jenny2/sample_cifar/test_seed_tuning_3"



# Declare an array of string with type
declare -a StringArray=("1" "300" "542" "23" "735" "91823" "273" "576" "690" "106")

path_sample_final_ant="$path_sample/$cfg"
path_sample_final_ant+="_c_$c"
path_sample_final_ant+="_seed_$seed_train"
mkdir $path_sample_final_ant

# Iterate the string array using for loop
for val in ${StringArray[@]}; do
   path_sample_final=$path_sample_final_ant
   path_sample_final+="/$val"
#   #echo $path_sample_final
   python generate.py --pics 10000 --sample 1 --ckpt $ckpt_path --cfg $cfg --c $c --device $device --path_sample $path_sample_final --seed $val --activation_mix 0
done
