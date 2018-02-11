#!/bin/bash
#Usage: ./split_embedding.sh folder_name

folder=$1
iter=$2
testratio=$3
embiter=$4
addmode=$5
param=$6

for file in ./graph/$folder/*.edgelist
do
out="$folder"_"$iter"_"$testratio"_"$embiter"_"$addmode"_"$param"

echo working on $file >> $out.txt
# python src/main_link.py --input $file --weighted --num-walks 2 --segment 1 --score-iter $iter --popwalk none --iter 50 >> $out.txt
python src/main_link.py --input $file --num-walks 2 --segment 1 --score-iter $iter --popwalk none --iter $embiter --add-user-edges --unseparated >> $out.txt
python src/main_link.py --input $file --num-walks 2 --segment 1 --score-iter $iter --popwalk none --iter $embiter --add-user-edges --user-edges-mode $addmode --user-edges-ratio 0.10 --unseparated >> $out.txt
python src/main_link.py --input $file --num-walks 2 --segment 1 --score-iter $iter --popwalk none --iter $embiter --add-user-edges --user-edges-mode $addmode --user-edges-ratio 0.20 --unseparated >> $out.txt
python src/main_link.py --input $file --num-walks 2 --segment 1 --score-iter $iter --popwalk none --iter $embiter --add-user-edges --user-edges-mode $addmode --user-edges-ratio 0.50 --unseparated >> $out.txt
python src/main_link.py --input $file --num-walks 2 --segment 1 --score-iter $iter --popwalk none --iter $embiter --add-user-edges --user-edges-mode $addmode --user-edges-ratio 0.80 --unseparated >> $out.txt
done
exit 0