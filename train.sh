set -u
set -e

source config

data=data
lexicon=seed-lexicon
output=word-vec

./bin/embeddingMatching -mono-train1 data/$config/$data.$lang1 -mono-train2 data/$config/$data.$lang2 -lexicon1 data/$config/$lexicon.$lang1 -lexicon2 data/$config/$lexicon.$lang2 -output1 data/$config/$output.$lang1 -output2 data/$config/$output.$lang2 -min-count 1 -size 40 -window 5 -sample 1e-5 -negative 5 -threshold 0.5 -epochs 10 -threads 1 -adagrad 0 -lexicon-lambda 0.01 -matching-lambda 1000 -alpha 0.1
