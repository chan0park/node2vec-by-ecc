#!/bin/bash
#Usage: ./split_embedding.sh folder_name

folder=$1

for file in ./graph/$folder/*.edgelist
do
out="${file%.*}"
out="${out##*/}"
echo working on $file, $out
python src/main.py --input $file --output ./emb/$folder/$out.emb --weighted --num-walks 3 --walk-length 40
done
exit 0