#!/bin/bash
#Usage: ./split_embedding.sh folder_name

folder=$1
scoreiter=$2
testratio=$3
numwalk=$4

for file in ./graph/$folder/*.edgelist
do
out="$folder"_"$iter"_"$testratio"_"$numwalk"_100500

echo working on $file >> auc_results/$out.txt
python src/main_link.py --input $file --num-walks $numwalk --segment 5 --score-iter $scoreiter --popwalk none --iter 100 --test-ratio $testratio --unseparated >> auc_results/$out.txt
python src/main_link.py --input $file --num-walks $numwalk --segment 5 --score-iter $scoreiter --popwalk none --iter 500 --test-ratio $testratio --unseparated >> auc_results/$out.txt
# python src/main_link.py --input $file --weighted --num-walks 2 --segment 1 --score-iter $iter --popwalk pop >> $out.txt
# python src/main_link.py --input $file --weighted --num-walks 2 --segment 1 --score-iter $iter --popwalk both >> $out.txt
done
exit 0