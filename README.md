# Faithfulness 😇

An easy-to-use library to evaluate faithfulness (factual correctness) of abstractive summaries. Faithfulness is computed by comparing a summary with its original source document.

This library includes multiple faithfulness metrics based on:
- BERTScore
- Entailment 
- Question Generation & Question Answering framework (QGQA)
- Named Entity Recognition (NER)
- Open Information Extraction (OpenIE)
- Semantic Role Labeling (SRL)
- Sentence Similarity (SentSim)

## Installation ⚙️

1. `$ conda create -n my_project python=3.8` This creates a new virtual environment for your project with conda. You can activate it with `$ conda activate my_project`.
2. `$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch` Please install PyTorch by following the instructions [here](https://pytorch.org/get-started/locally/). Make sure to install the CUDA variant that matches the CUDA version of your GPU. 
3. `$ pip install faithfulness` This installs the faithfulness library and it's dependencies. Read more about the dependencies [below](#dependencies-).

All faithfulness metrics are model-based. Some models have to be installed manually:
- Download the SRL model [here](https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz) and save it in your project. e.g. __/models/srl_model.tar.gz__
- Download a spacy model: `$ python -m spacy download en_core_web_sm`
- Download CoreNLP: `import stanza && stanza.install_corenlp()`

## Usage 🔥
```
from faithfulness.QGQA import QGQA

qgqa = QGQA()
summary = "Lorem ipsum dolor sit amet"
source = "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam ..."
result: QGQAResult = qgqa.score(summary, source)
print(f"Faithfulness: {result["f1"]}")
```

More examples can be found [here 💯](https://github.com/bigabig/faithfulness/examples/).

## Evaluation 📊
We evaluated all faithfulness metrics by correlating them with human judgements on the XSUM dataset ([link](https://github.com/google-research-datasets/xsum_hallucination_annotations)).
You will soon be able to read more about the evaluation in our paper. ([Master's thesis](https://www.inf.uni-hamburg.de/en/inst/ab/lt/teaching/theses/completed-theses/2021-ma-timfischer.pdf))

| Method     | Pearson (r) | Spearman (p) |
|------------|-------------|--------------|
| 🥇 BERTScore  | 0.501       | 0.486        |
| 🥈 Entailment | 0.366       | 0.422        |
| 🥉 SentSim    | 0.392       | 0.389        |
| SRL        | 0.393       | 0.377        |
| NER        | 0.252       | 0.259        |
| QGQA       | 0.228       | 0.258        |
| OpenIE     | 0.169       | 0.185        |

### Reproduce results & evaluate custom dataset
You can download the preprocessed XSUM dataset [here](https://data.bigabig.de/prepared_xsum.json) and the preprocessed Summeval dataset [here](https://data.bigabig.de/prepared_summeval.json) to reproduce the above results.
To evaluate the faithfulness metrics on other datasets, we recommend using the provided Experimentor class. For this, your dataset has to be in the following JSON format:

prepared_dataset.json
```
[{
    "summary": "the summary text..."
    "source": "the source text..."
    "summary_sentences": ["summary sentence 1", "summary sentence 2", ...] 
    "source_sentences": ["source sentence 1", "source sentence 2", ...] 
    "faithfulness": 0.0 - 1.0
}, ...]
```

You can now use the experimentor:
```
output_path=Path("./experiments/dataset/qgqa"
faithfulness_metric = QGQA(metric=BERTScore, save_path=output_path, batch_mode=True)
Experimentor(data_path=Path("./prepared_dataset.json"),
             out_path=output_path),
             metric=faithfulness_metric,
             experiment_name="qgqa_bertscore").experiment()
```
In the above example, correlations are written to /experiments/dataset/qgqa/qgqa_bertscore.csv


## Dependencies 🔗

By running `$ pip install faithfulness` you will install this library as well as the following dependencies:
- [🤗 transformers](https://huggingface.co/transformers/)
- [spaCy](https://spacy.io/) (used for Entailment, NER, QGQA, SentSim, SRL)
- [Stanza](https://stanfordnlp.github.io/stanza/) (used for OpenIE)
- [AllenNLP](https://allennlp.org/) (used for SRL)
- [SentenceTransformers](https://www.sbert.net/) (used for NER, OpenIE, QGQA, SentSim, SRL)

## Troubleshooting 🛠
There are currently problems when installing allennlp and jsonnet. If you encounter "Building wheel for jsonnet (setup.py) ... error_" during the installation please try:
```
apt-get install make 
apt-get install g++ 
```
or install jsonnet before installing this library
```
conda install -c conda-forge jsonnet
pip install faithfulness
```
