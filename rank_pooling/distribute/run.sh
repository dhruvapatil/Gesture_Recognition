#!/bin/bash

machine_name=$1
list_index=$2
echo $machine_name
#ssh -x $machine_name "(cd /s/chopin/k/grad/dkpatil/PycharmProjects/ChaLearn/studies/; nohup nice -n 19 python -u train_rankpooling.py \"$list_index\" 2>>error.log) &" &
ssh -x $machine_name "(cd /s/chopin/k/grad/dkpatil/PycharmProjects/ChaLearn/studies/; nohup nice -n 19 python -u val_rankpooling.py \"$list_index\" 2>>error.log) &" &
#ssh -x $machine_name "(cd /s/chopin/k/grad/dkpatil/PycharmProjects/ChaLearn/studies/; nohup nice -n 19 python -u valid_rankpooling.py \"$list_index\" 2>>error.log) &" &