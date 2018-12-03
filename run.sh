set -u
set -e

./train.sh
./translate.sh word-vec vocab result

