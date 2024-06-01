for net in 3x100; do
for seed in -1; do
for ndims in 5; do
for eps in 0.1; do
for pick in grouped_block; do
for num in 1 2 3 4 5 6 7 8 9; do
    time \
    python3 ./eval_6_lookup.py \
        --net=${net} \
        --num=${num} \
        --ndims=${ndims} \
        --eps=${eps} \
        --seed=${seed} \
        --pick=${pick} \
        --device='cuda:1' \
    | tee -a log_eval_6_lookup_${net}_${num}p_ndims=${ndims}_eps=${eps}_pick=${pick}_k=${k}_seed=${seed}.txt
done
done
done
done
done
done
