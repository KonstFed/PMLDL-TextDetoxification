# PMLDL-TextDetoxification

## Dependecies

You can use exact requirements as:

```bash
pip3 install -r requirements_freeze.txt
```

For broad requirements use:

```bash
pip3 install -r requirements.txt
```

## Weights

You can download weights from [here](https://disk.yandex.com/d/YIPCDBISwf6cwQ). There would be 3 archives:
- parahrasing.zip: detoxification model itself
- preprocessing.zip: preprocessing for logistic regression
- toxicity_cl.zip: weights for logistic regression and DistilBert

Extract all files to folder __models__

final structure should be
```bash
models
├── paraphrasing
├── preprocessing
└── toxicity_cl
```

## Data

To load data:
```bash
mkdir -p data
cd data
wget https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip
unzip filtered.tsv
cd ..
```

To split data into train and test tsv

```bash
python3 src/data/make_dataset.py
```

## Inference

### Classification
Example of usage for sentence toxicity classification.
```bash
python3 -m src.inference -c configs/sentence_toxic_cls/hard_labels.yaml "I love cats"
```

For this task following configs works:
- [distilbert](configs/sentence_toxic_cls/distilbert_inference.yaml)
- [logistic regression for soft labels, weighted BoW](configs/sentence_toxic_cls/soft_labels.yaml)
- [logistic regression for hard labels, weighted BoW](configs/sentence_toxic_cls/hard_labels.yaml)

### Detoxification

```bash
python3 -m src.inference -c configs/parahrasing/t5baseline_inference.yaml
```

## Train
Example of usage is the same for both tasks.
```bash
python3 -m src.train -c configs/sentence_toxic_cls/hard_labels.yaml
```

## Test
Test dataset using metric and produce result in .tsv file
```
python3 -m src.test -b 32 -t configs/sentence_toxic_cls/distilbert_inference.yaml -o ./tmp -c configs/parahrasing/t5baseline.yaml data/test.tsv
```

