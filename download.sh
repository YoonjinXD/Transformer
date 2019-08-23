mkdir data
mkdir data/translation
mkdir data/translation/train
mkdir data/translation/test

wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en data/translation/train
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de data/translation/train
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2012.en data/translation/test
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2012.de data/translation/test

echo "Data has downloaded."

