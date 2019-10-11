#!/bin/bash

for t in 1 2 3; do
for h in 1 4 16 64; do
    b=160
    r=0.001
    m=0
    a=sigmoid
    wd=$h-$r-$b-$m-$a-$t
    mkdir -p $wd
    ./pinniped.py --train-arff datasets/Phoneme/Phoneme_TRAIN.arff --test-arff datasets/Phoneme/Phoneme_TEST.arff \
		  --layer-dims 1024,$h,39 --activation-unit $a --reserve-every=4 --learning-rate=$r --batch-size=$b --momentum $m --sgd \
		  --plot-every=256 --plot-confusion-every=8 --epochs=256 --workspace-dir=$wd --autograd-backprop --save-every=1 2> $wd/log
done

for r in 0.1 0.01 0.001; do
    b=160
    h=64
    m=0
    a=sigmoid
    wd=$h-$r-$b-$m-$a-$t
    mkdir -p $wd
    ./pinniped.py --train-arff datasets/Phoneme/Phoneme_TRAIN.arff --test-arff datasets/Phoneme/Phoneme_TEST.arff \
    		  --layer-dims 1024,$h,39 --activation-unit $a --reserve-every=4 --learning-rate=$r --batch-size=$b --momentum $m --sgd \
		  --plot-every=256 --plot-confusion-every=8 --epochs=256 --workspace-dir=$wd --autograd-backprop --save-every=1 2> $wd/log
done

for b in 1 40 80 160; do
    h=64
    r=0.001
    m=0
    a=sigmoid
    wd=$h-$r-$b-$m-$a-$t
    mkdir -p $wd
    ./pinniped.py --train-arff datasets/Phoneme/Phoneme_TRAIN.arff --test-arff datasets/Phoneme/Phoneme_TEST.arff \
		  --layer-dims 1024,$h,39 --activation-unit $a --reserve-every=4 --learning-rate=$r --batch-size=$b --momentum $m --sgd \
		  --plot-every=256 --plot-confusion-every=8 --epochs=256 --workspace-dir=$wd --autograd-backprop --save-every=1 2> $wd/log
done

for m in 0.1 0.5 0.9 1; do
    b=160
    h=64
    r=0.001
    a=sigmoid
    wd=$h-$r-$b-$m-$a-$t
    mkdir -p $wd
    ./pinniped.py --train-arff datasets/Phoneme/Phoneme_TRAIN.arff --test-arff datasets/Phoneme/Phoneme_TEST.arff \
		  --layer-dims 1024,$h,39 --activation-unit $a --reserve-every=4 --learning-rate=$r --batch-size=$b --momentum $m --sgd \
		  --plot-every=256 --plot-confusion-every=8 --epochs=256 --workspace-dir=$wd --autograd-backprop --save-every=1 2> $wd/log
done

for a in tanh relu; do
    b=160
    h=64
    r=0.001
    m=0
    wd=$h-$r-$b-$m-$a-$t
    mkdir -p $wd
    ./pinniped.py --train-arff datasets/Phoneme/Phoneme_TRAIN.arff --test-arff datasets/Phoneme/Phoneme_TEST.arff \
		  --layer-dims 1024,$h,39 --activation-unit $a --reserve-every=4 --learning-rate=$r --batch-size=$b --momentum $m --sgd \
		  --plot-every=256 --plot-confusion-every=8 --epochs=256 --workspace-dir=$wd --autograd-backprop --save-every=1 2> $wd/log
done

done
