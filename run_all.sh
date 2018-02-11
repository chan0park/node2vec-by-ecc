#!/bin/bash
#Usage: ./split_embedding.sh folder_name

folder=$1
iter=$2
testratio=$3
embiter=$4

for file in ./graph/$folder/*.edgelist
do
out="$folder"_"$iter"_"$testratio"_"$embiter"

echo working on $file >> $out.txt
python src/main_link.py --input $file --num-walks 10 --segment 1 --score-iter $iter --popwalk none --iter 100 --test-ratio $testratio --unseparated >> $out.txt
python src/main_link.py --input $file --num-walks 10 --segment 1 --score-iter $iter --popwalk none --iter 250 --test-ratio $testratio --unseparated >> $out.txt
python src/main_link.py --input $file --num-walks 10 --segment 1 --score-iter $iter --popwalk none --iter 500 --test-ratio $testratio --unseparated >> $out.txt
done
exit 0