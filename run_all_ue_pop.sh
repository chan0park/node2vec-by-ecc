#!/bin/bash
#Usage: ./split_embedding.sh folder_name

folder=$1
iter=$2
testratio=$3

for file in ./graph/$folder/*.edgelist
do
out="$folder"
out+="_"
out+=$iter
out+="_"
out+=$testratio
out+="_embiter50_numwalk10_pop"
echo working on $file >> $out.txt
python src/main_link.py --input $file --weighted --num-walks 10 --segment 1 --score-iter $iter --popwalk none --iter 50 >> $out.txt
python src/main_link.py --input $file --weighted --num-walks 10 --segment 1 --score-iter $iter --popwalk pop --iter 50 >> $out.txt
python src/main_link.py --input $file --weighted --num-walks 10 --segment 1 --score-iter $iter --popwalk both --iter 50 >> $out.txt
done
exit 0