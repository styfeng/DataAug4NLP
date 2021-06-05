# Data Augmentation Techniques for NLP 


If you'd like to add your paper, do not email us. Instead, read the protocol for [adding a new entry](https://github.com/styfeng/DataAug4NLP/blob/main/rules.md) and send a pull request.

We group the papers by [text classification](#text-classification), [translation](#translation), [summarization](#summarization), [question-answering](#question-answering), [sequence tagging](#sequence-tagging), [parsing](#parsing), [grammatical-error-correction](#grammatical-error-correction), [generation](#generation), [dialogue](#dialogue), [multimodal](#multimodal), [mitigating bias](#mitigating-bias), [mitigating class imbalance](#mitigating-class-imbalance), and [adversarial examples](#adversarial-examples).

This repository is based on our paper, ["A survey of data augmentation approaches in NLP (Findings of ACL '21)"](http://arxiv.org/abs/2105.03075). You can cite it as follows:
```
@article{feng2021survey,
  title={A Survey of Data Augmentation Approaches for NLP},
  author={Feng, Steven Y and Gangal, Varun and Wei, Jason and Chandar, Sarath and Vosoughi, Soroush and Mitamura, Teruko and Hovy, Eduard},
  journal={Findings of ACL},
  year={2021}
}
```
Authors: <a href="https://scholar.google.ca/citations?hl=en&user=zwiszZIAAAAJ">Steven Y. Feng</a>,
			  <a href="https://scholar.google.com/citations?user=rWZq2nQAAAAJ&hl=en">Varun Gangal</a>,
			  <a href="https://scholar.google.com/citations?user=wA5TK_0AAAAJ&hl=en">Jason Wei</a>,
			  <a href="https://scholar.google.co.in/citations?user=yxWtZLAAAAAJ&hl=en">Sarath Chandar</a>,
			  <a href="https://scholar.google.ca/citations?user=45DAXkwAAAAJ&hl=en">Soroush Vosoughi</a>,
			  <a href="https://scholar.google.com/citations?user=gjsxBCkAAAAJ&hl=en">Teruko Mitamura</a>,
			  <a href="https://scholar.google.com/citations?user=PUFxrroAAAAJ&hl=en">Eduard Hovy</a>

Note: WIP. More papers will be added from our survey paper to this repo over the next month or so.

Inquiries should be directed to stevenyfeng@gmail.com or by opening an issue here.

### Text Classification
| Paper | Datasets | 
| -- | --- |
| Unsupervised Word Sense Disambiguation Rivaling Supervised Methods , ([ACL '95](https://www.aclweb.org/anthology/P95-1026.pdf)) | Paper-Specific/Legacy Corpus | 
| Synonym Replacement (Character-Level Convolutional Networks for Text Classification, [NeurIPS '15](https://papers.nips.cc/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf)) | AG’s News, DBPedia, Yelp, Yahoo Answers, Amazon | 
| That’s So Annoying!!!: A Lexical and Frame-Semantic Embedding Based Data Augmentation Approach to Automatic Categorization of Annoying Behaviors using #petpeeve Tweets [(EMNLP '15)](https://www.aclweb.org/anthology/D15-1306.pdf) | twitter| 
| Robust Training under Linguistic Adversity [(EACL '17)](https://www.aclweb.org/anthology/E17-2004/) [code](https://github.com/lrank/Linguistic_adversity) | Movie review, customer review, SUBJ, SST | 
| Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations [(NAACL '18)](https://www.aclweb.org/anthology/N18-2072.pdf) [code](https://github.com/pfnet-research/contextual_augmentation) | SST, SUBJ, MRQA, RT, TREC | 
| Variational Pretraining for Semi-supervised Text Classification [(ACL '19)](https://www.aclweb.org/anthology/P19-1590.pdf) [code](http://github.com/allenai/vampire) | IMDB, AG News, Yahoo, hatespeech | 
| EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks [(EMNLP '19)](http://dx.doi.org/10.18653/v1/D19-1670) [code](https://github.com/jasonwei20/eda_nlp) | SST, CR, SUBJ, TREC, PC |
| Nonlinear Mixup: Out-Of-Manifold Data Augmentation for Text Classification [(AAAI '20)](https://doi.org/10.1609/aaai.v34i04.5822) | TREC, SST, Subj, MR |
| MixText: Linguistically-Informed Interpolation of Hidden Space for Semi-Supervised Text Classification [(ACL '20)](https://www.aclweb.org/anthology/2020.acl-main.194/) [code](https://github.com/GT-SALT/MixText) | AG News, DBpedia, Yahoo, IMDb | 
| Unsupervised Data Augmentation for Consistency Training [(NeurIPS '20)](https://papers.nips.cc/paper/2020/hash/44feb0096faa8326192570788b38c1d1-Abstract.html) [code](https://papers.nips.cc/paper/2020/hash/44feb0096faa8326192570788b38c1d1-Abstract.html) | Yelp, IMDb, amazon, DBpedia | 
| Not Enough Data? Deep Learning to the Rescue! [(AAAI '20)](https://arxiv.org/abs/1911.03118) | ATIS, TREC, WVA | 
| SSMBA: Self-Supervised Manifold Based Data Augmentation for Improving Out-of-Domain Robustness [(EMNLP '20)](https://www.aclweb.org/anthology/2020.emnlp-main.97/) [code](https://github.com/nng555/ssmba) | IWSLT'14 | 
| Data Boost: Text Data Augmentation Through Reinforcement Learning Guided Conditional Generation [(EMNLP '20)](https://www.aclweb.org/anthology/2020.emnlp-main.726/) | ICWSM 20’ Data Challenge, SemEval '17 sentiment analysis, SemEval '18 irony |
| Textual Data Augmentation for Efficient Active Learning on Tiny Datasets [(EMNLP '20)](https://www.aclweb.org/anthology/2020.emnlp-main.600/) | SST2, TREC |
| Text Augmentation in a Multi-Task View [(EACL '21)](https://www.aclweb.org/anthology/2021.eacl-main.252/) | SST2, TREC, SUBJ | 
| Few-Shot Text Classification with Triplet Loss, Data Augmentation, and Curriculum Learning [(NAACL '21)](https://arxiv.org/abs/2103.07552) [code](https://github.com/jasonwei20/triplet-loss) | HUFF, COV-Q, AMZN, FEWREL | 

### Natural Language Generation

| Paper | Datasets | 
| -- | --- |
| GenAug: Data Augmentation for Finetuning Text Generators [(DeeLIO @ EMNLP '20)](https://www.aclweb.org/anthology/2020.deelio-1.4/) [code](https://github.com/styfeng/GenAug) | TO-DO | 

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
| Data Augmentation for BERT Fine-Tuning in Open-Domain Question Answering [(arxiv '19)](https://arxiv.org/abs/1904.06652) | SQuAD, Trivia-QA, CMRC, DRCD | 
| XLDA: Cross-Lingual Data Augmentation for Natural Language Inference and Question Answering [(arxiv '19)](https://openreview.net/forum?id=BJgAf6Etwr) | XNLI, SQuAD |
| Synthetic Data Augmentation for Zero-Shot Cross-Lingual Question Answering [(arxiv '20)](https://arxiv.org/abs/2010.12643) | MLQA, XQuAD, SQuAD-it, PIAF | 
| Logic-Guided Data Augmentation and Regularization for Consistent Question Answering [(ACL '20)](https://www.aclweb.org/anthology/2020.acl-main.499/) [code](https://github.com/AkariAsai/logic_guided_qa) | WIQA, QuaRel, HotpotQA | 

### Summarization

| Paper | Datasets | 
| -- | --- |
| Transforming Wikipedia into Augmented Data for Query-Focused Summarization [(arxiv '19)](https://arxiv.org/abs/1911.03324) | DUC |
| Iterative Data Augmentation with Synthetic Data (Abstract Text Summarization: A Low Resource Challenge [(EMNLP '19)](https://www.aclweb.org/anthology/D19-1616/) | Swisstext, commoncrawl | 
| Improving Zero and Few-Shot Abstractive Summarization with Intermediate Fine-tuning and Data Augmentation [(NAACL '21)](https://arxiv.org/abs/2010.12836) | CNN-DailyMail | 


### Sequence Tagging

| Paper | Datasets | 
| -- | --- |
| Data Augmentation via Dependency Tree Morphing for Low-Resource Languages [(EMNLP '18)](https://www.aclweb.org/anthology/D18-1545.pdf) [code](https://github.com/gozdesahin/crop-rotate-augment) | universal dependencies project | 

### Parsing
TODO: https://www.aclweb.org/anthology/2020.emnlp-main.107/

### Grammatical Error Correction
| Paper | Datasets | 
| -- | --- |
| Using  Wikipedia  Edits  in  Low Resource Grammatical Error Correction.   [(WNUT @ EMNLP '18)](https://doi.org/10.18653/v1/W18-6111) | Falko-MERLIN GEC Corpus |
| Sequence-to-sequence Pre-training with Data Augmentation for Sentence Rewriting   [(arxiv '19)](https://arxiv.org/abs/1909.06002) | CoNLL-2014 , JFLEG  |
| SwitchOut: an Efficient Data Augmentation Algorithm for Neural Machine Translation   [(EMNLP '18)](https://www.aclweb.org/anthology/D18-1100.pdf) | IWSLT 16 en-vi, IWSLT 15 de-en, WMT en-de |
| Neural Grammatical Error Correction  Systems  with  Unsupervised  Pre-training on Synthetic Data. [(BEA @ ACL '19)](https://doi.org/10.18653/v1/W19-4427) | FCE, NUCLE, W&I+LOCNESS, Lang-8 (BEA @ ACL '19 Shared Task) |
| A neural grammatical error cor-rection  system  built  on  better  pre-training  and  se-quential  transfer  learning. [(BEA @ ACL '19)](https://doi.org/10.18653/v1/W19-4423) | FCE, NUCLE, W&I+LOCNESS, Lang-8 (BEA @ ACL '19 Shared Task), Gutenberg, Tatoeba, WikiText-103 (Pretraining) |
| Improving  Grammatical  Error  Correction with  Data  Augmentation  by  Editing  Latent  Representation [(COLING'20)](https://doi.org/10.18653/v1/2020.coling-main.200) | FCE, NUCLE, W&I+LOCNESS, Lang-8 (BEA @ ACL '19 Shared Task)  |
| Noising and Denoising Natural Language:  Diverse Backtranslation for Grammar  Correction. [(NAACL'18)](https://www.aclweb.org/anthology/N18-1057/)  | Lang-8, CoNLL-2014, CoNLL-2013, JFLEG |
| Corpora Generation for Grammatical Error Correction [(NAACL'19)](https://doi.org/10.18653/v1/N19-1333)  | CoNLL-2014, JFLEG, Lang-8 |

### Dialogue
TO-DO

### Multimodal
TO-DO

### Mitigating Bias
TO-DO

### Mitigating Class Imbalance
TO-DO

### Adversarial examples

| Paper | Datsets | 
| -- | --- |
| Adversarial Example Generation with Syntactically Controlled Paraphrase Networks [(NAACL '18)](https://www.aclweb.org/anthology/N18-1170/) | SST, SICK | 
Make sure we get textattack

### Compositionality

| Paper | Datsets | 
| -- | --- |
| Good-Enough Compositional Data Augmentation [(ACL '20)](https://www.aclweb.org/anthology/2020.acl-main.676.pdf) [code](https://github.com/jacobandreas/geca) | SCAN |
| Sequence-Level Mixed Sample Data Augmentation [(EMNLP '20)](https://www.aclweb.org/anthology/2020.emnlp-main.447) [code](https://github.com/dguo98/seqmix) | SCAN | 

### Popular Resources
- [A visual survey of data augmentation in NLP](https://amitness.com/2020/05/data-augmentation-for-nlp/)
- [nlpaug](https://github.com/makcedward/nlpaug)
