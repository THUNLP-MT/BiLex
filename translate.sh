set -u
set -e

source config

file=$1
toTranslate=$2
output=$3

python3 scripts/processBilbowa.py data/$config/$file.$lang1 data/$config/vocab.$lang1 data/$config/vec.$lang1
python3 scripts/processBilbowa.py data/$config/$file.$lang2 data/$config/vocab.$lang2 data/$config/vec.$lang2

python3 scripts/translate.py data/$config/$file.$lang1 data/$config/$file.$lang2 data/$config/$toTranslate.$lang1 data/$config/$output
