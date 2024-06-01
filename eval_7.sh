for net in 3x100
do
for seed in -1 0 1 2 3 4 5
do
for img_seed in 0 1 2 3 4 5 6 7 8 9 10
do
for pick in \
    leading \
    center \
    nonzero \
    random \
    grouped \
    grouped_block \
    grouped_row
do
for eps in 0.1
do
for ndims in 5 6 7 8 9 10 11 12 13 14 15 16
do
for k in 2
do
    python3 ./eval_7_aprnn.py \
        --net=${net} \
        --ndims=${ndims} \
        --eps=${eps} \
        --seed=${seed} \
        --pick=${pick} \
        --k=${k} \
        --img_seed=${img_seed} \
        --device='cuda:0' \
    | tee -a log_2_eval_6_aprnn_${net}_ndims=${ndims}_eps=${eps}_pick=${pick}_k=${k}_seed=${seed}_img_seed=${img_seed}.txt
done
done
done
done
done
done
done

