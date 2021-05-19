# Data Augmentation Techniques for NLP 


If you'd like to add your paper, do not email us. Instead, read the protocol for [adding a new entry](https://github.com/styfeng/DataAug4NLP/blob/main/rules.md) and send a pull request.

We group the papers by [text classification](#text-classification), [translation](#translation), [summarization](#summarization), [question-answering](#question-answering), [sequence tagging](#sequence-tagging), [parsing](#parsing), [grammatical-error-correction](#grammatical-error-correction), [generation](#generation), [dialogue](#dialogue), [multimodal](#multimodal), [few-shot learning](#few-shot-learning), [mitigating bias](#mitigating-bias), [mitigating class imbalance](#mitigating-class-imbalance), and [adversarial examples](#adversarial-examples).

This repository is based on our paper, ["A survey of data augmentation approaches in NLP (Findings of ACL '21)"](http://arxiv.org/abs/2105.03075). You can cite it as follows:
```
@article{feng2021survey,
  title={A Survey of Data Augmentation Approaches for NLP},
  author={Feng, Steven Y and Gangal, Varun and Wei, Jason and Chandar, Sarath and Vosoughi, Soroush and Mitamura, Teruko and Hovy, Eduard},
  journal={Findings of ACL},
  year={2021}
}
```

### Text Classification
| Paper | Datasets | 
| -- | --- |
| Synonym Replacement (Character-level convolutional networks for text classification, [NeurIPS '15](https://papers.nips.cc/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf)) | AG’s News, DBPedia, Yelp, Yahoo Answers, Amazon | 
| Robust training under linguistic adversity [(EACL '17)](https://www.aclweb.org/anthology/E17-2004/) [code](https://github.com/lrank/Linguistic_adversity) | Movie review, customer review, SUBJ, SST | 
| Unsupervised data augmentation for consistency training [(NeurIPS '20)](https://papers.nips.cc/paper/2020/hash/44feb0096faa8326192570788b38c1d1-Abstract.html) [code](https://papers.nips.cc/paper/2020/hash/44feb0096faa8326192570788b38c1d1-Abstract.html) | Yelp, IMDb, amazon, DBpedia | 
| Nonlinear Mixup: Out-Of-Manifold Data Augmentation for Text Classification [(AAAI '20)](https://doi.org/10.1609/aaai.v34i04.5822) | TREC, SST, Subj, MR |
| MixText: Linguistically-Informed Interpolation of Hidden Space for Semi-Supervised Text Classification [(ACL '20)](https://www.aclweb.org/anthology/2020.acl-main.194/) [code](https://github.com/GT-SALT/MixText) | AG News, DBpedia, Yahoo, IMDb | 
| Variational Pretraining for Semi-supervised Text Classification [(ACL '19)](https://www.aclweb.org/anthology/P19-1590.pdf) [code](http://github.com/allenai/vampire) | IMDB, AG News, Yahoo, hatespeech | 
| Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations [(NAACL '18)](https://www.aclweb.org/anthology/N18-2072.pdf) [code](https://github.com/pfnet-research/contextual_augmentation) | SST, SUBJ, MRQA, RT, TREC | 
| SSMBA: Self-Supervised Manifold Based Data Augmentation for Improving Out-of-Domain Robustness [(EMNLP '20)](https://www.aclweb.org/anthology/2020.emnlp-main.97/) [code](https://github.com/nng555/ssmba) | IWSLT'14 | 
| Not Enough Data? Deep Learning to the Rescue! [(AAAI '20)](https://arxiv.org/abs/1911.03118) | ATIS, TREC, WVA | 
| Textual Data Augmentation for Efficient Active Learning on Tiny Datasets [(EMNLP '20)](https://www.aclweb.org/anthology/2020.emnlp-main.600/) | SST2, TREC |

### Translation

| Paper | Datasets | 
| -- | --- |
| Backtranslation (Improving Neural Machine Translation Models with Monolingual Data, [ACL '16](https://www.aclweb.org/anthology/P16-1009.pdf)) | WMT '15 en-de, IWSLT ''15 en-tr |
| SwitchOut: an Efficient Data Augmentation Algorithm for Neural Machine Translation [(EMNLP '18)](https://www.aclweb.org/anthology/D18-1100/) | IWSLT '15 en-vi, IWSLT '16 de-en, WMT '15 en-de |
| Soft Contextual Data Augmentation for Neural Machine Translation [(ACL '19)](https://www.aclweb.org/anthology/P19-1555/) [code](https://github.com/teslacool/SCA) | IWSLT '14 de/es/he-en, WMT '14 en-de | SSMBA: Self-Supervised Manifold Based Data Augmentation for Improving Out-of-Domain Robustness [(EMNLP '20)](https://www.aclweb.org/anthology/2020.emnlp-main.97/) [code](https://github.com/nng555/ssmba) | IWSLT'14 | 


### Question Answering

| Paper | Datasets | 
| -- | --- |
| An Exploration of Data Augmentation and Sampling Techniques for Domain-Agnostic Question Answering [(EMNLP '19 Workshop)](https://www.aclweb.org/anthology/D19-5829/) | MRQA| 
| Data Augmentation for BERT Fine-Tuning in Open-Domain Question Answering [(arxiv)](https://arxiv.org/abs/1904.06652) | SQuAD, Trivia-QA, CMRC, DRCD | 
| Synthetic Data Augmentation for Zero-Shot Cross-Lingual Question Answering [(arxiv)](https://arxiv.org/abs/2010.12643) | MLQA, XQuAD, SQuAD-it, PIAF | 
| XLDA: Cross-Lingual Data Augmentation for Natural Language Inference and Question Answering [(arxiv)](https://openreview.net/forum?id=BJgAf6Etwr) | XNLI, SQuAD | Logic-Guided Data Augmentation and Regularization for Consistent Question Answering [(ACL '20)](https://www.aclweb.org/anthology/2020.acl-main.499/)[code](https://github.com/AkariAsai/logic_guided_qa) | WIQA, QuaRel, HotpotQA | 

### Summarization

| Paper | Datasets | 
| -- | --- |
| Improving Zero and Few-Shot Abstractive Summarization with Intermediate Fine-tuning and Data Augmentation [(NAACL '21)](https://arxiv.org/abs/2010.12836) | CNN-DailyMail | 
| Iterative Data Augmentation with Synthetic Data (Abstract Text Summarization: A Low Resource Challenge [(EMNLP '19)](https://www.aclweb.org/anthology/D19-1616/) | Swisstext, commoncrawl | 
| Transforming Wikipedia into Augmented Data for Query-Focused Summarization [(arxiv)](https://arxiv.org/abs/1911.03324) | DUC |


### Sequence Tagging

| Paper | Datasets | 
| -- | --- |
| Data Augmentation via Dependency Tree Morphing for Low-Resource Languages [(EMNLP '18)](https://www.aclweb.org/anthology/D18-1545.pdf) [code](https://github.com/gozdesahin/crop-rotate-augment) | universal dependencies project | 

### Parsing
https://www.aclweb.org/anthology/2020.emnlp-main.107/ (lol does this fall under parsing? -jason)

### Grammatical Error Correction

### Generation

### Dialogue

### Multimodal

### Few-shot learning

### Mitigating Bias

### Mitigating Class Imbalance

### Adversarial examples

| Paper | Datsets | 
| -- | --- |
| Adversarial Example Generation with Syntactically Controlled Paraphrase Networks [(NAACL '18)](https://www.aclweb.org/anthology/N18-1170/) | SST, SICK | 

### Compositionality

| Paper | Datsets | 
| -- | --- |
| Good-enough compositional data augmentation [(ACL '20)](https://www.aclweb.org/anthology/2020.acl-main.676.pdf) [code](https://github.com/jacobandreas/geca) | SCAN |
| Sequence-level mixed sample data augmentation [(EMNLP '20)](https://www.aclweb.org/anthology/2020.emnlp-main.447) [code](https://github.com/dguo98/seqmix) | SCAN |

### Papers by the authors of this repository

To provide an unbiased list of work, we do not include our own work above and instead show it below: 

- Keep calm and switch on! Preserving sentiment and fluency in semantic text exchange [(EMNLP '19)](https://www.aclweb.org/anthology/D19-1272/)
- EDA: Easy Data Augmentation techniques for boosting performance on text classification tasks [(EMNLP '19)](http://dx.doi.org/10.18653/v1/D19-1670)
