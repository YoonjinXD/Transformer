mkdir data
mkdir data/translation
mkdir data/translation/train
mkdir data/translation/test

cd data/translation/train
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de
cd ../test
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2012.en
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2012.de

echo "Data has downloaded."
