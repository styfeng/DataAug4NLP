# Data Augmentation Techniques for NLP 


If you'd like to add your paper, do not email us. Instead, read the protocol for [adding a new entry](https://github.com/styfeng/DataAug4NLP/blob/main/rules.md) and send a pull request.

We group the papers by [text classification](#text-classification), [translation](#translation), [summarization](#summarization), [question-answering](#question-answering), [sequence tagging](#sequence-tagging), [parsing](#parsing), [grammatical-error-correction](#grammatical-error-correction), [generation](#generation), [dialogue](#dialogue), [multimodal](#multimodal), [mitigating bias](#mitigating-bias), [mitigating class imbalance](#mitigating-class-imbalance), [adversarial examples](#adversarial-examples), [compositionality](#compositionality), and [automated augmentation](#automated-augmentation).

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
| Unsupervised Word Sense Disambiguation Rivaling Supervised Methods ([ACL '95](https://www.aclweb.org/anthology/P95-1026.pdf)) | Paper-Specific/Legacy Corpus | 
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


### Translation

| Paper | Datasets | 
| -- | --- |
| Backtranslation (Improving Neural Machine Translation Models with Monolingual Data, [ACL '16](https://www.aclweb.org/anthology/P16-1009.pdf)) | WMT '15 en-de, IWSLT '15 en-tr |
| Adapting Neural Machine Translation with Parallel Synthetic Data [(WMT '17)](https://www.aclweb.org/anthology/W17-4714/) | COMMON, 1 Billion Words, dev2013, XRCE, IT, E-Com| 
| Data Augmentation for Low-Resource Neural Machine Translation [(ACL '17)](https://www.aclweb.org/anthology/P17-2090/) [code](https://github.com/marziehf/DataAugmentationNMT) | WMT '14/'15/'16 en-de/de-en| 
| Synthetic Data for Neural Machine Translation of Spoken-Dialects [(arxiv '17)](https://arxiv.org/abs/1707.00079) | LDC2012T09, OpenSubtitles-2013| 
| Multi-Source Neural Machine Translation with Data Augmentation [(IWSLT '18)](https://arxiv.org/abs/1810.06826) | TED Talks| 
| SwitchOut: an Efficient Data Augmentation Algorithm for Neural Machine Translation [(EMNLP '18)](https://www.aclweb.org/anthology/D18-1100/) | IWSLT '15 en-vi, IWSLT '16 de-en, WMT '15 en-de |
| Generalizing Back-Translation in Neural Machine Translation [(WMT '19)](https://www.aclweb.org/anthology/W19-5205/) | ed NewsCrawl2, WMT'18 de-en| 
| Neural Fuzzy Repair: Integrating Fuzzy Matches into Neural Machine Translation [(ACL '19)](https://www.aclweb.org/anthology/P19-1175/) | DGT-TM en-ml/en-hu| 
| Augmenting Neural Machine Translation with Knowledge Graphs [(arxiv '19)](https://arxiv.org/abs/1902.08816) | WMT '14 -'18| 
| Generalized Data Augmentation for Low-Resource Translation [(ACL '19)](https://www.aclweb.org/anthology/P19-1579/) [code](https://github.com/xiamengzhou/DataAugForLRL)| ENG-HRL-LRL, HRL-LRL | 
| Improving Robustness of Machine Translation with Synthetic Noise [(NAACL '19)](https://www.aclweb.org/anthology/N19-1190/) [code](https://github.com/MysteryVaibhav/robust_mtnt)| EP, TED, MTNT en-fr en-jpn| 
| Soft Contextual Data Augmentation for Neural Machine Translation [(ACL '19)](https://www.aclweb.org/anthology/P19-1555/) [code](https://github.com/teslacool/SCA) | IWSLT '14 de/es/he-en, WMT '14 en-de |
| Data augmentation using back-translation for context-aware neural machine translation [(DiscoMT @ EMNLP '19)](https://www.aclweb.org/anthology/D19-6504/) [code](https://github.com/sugi-a/discomt2019) | IWSLT'17 en-ja/en-fr, BookCorpus, Europarl v7, National Diet of Japan | 
| Improving Neural Machine Translation Robustness via Data Augmentation: Beyond Back-Translation [(W-NUT @ EMNLP '19)](https://www.aclweb.org/anthology/D19-5543/) | WMT'15/'19 en/fr, MTNT, IWSLT'17, MuST-C | 
| Data augmentation for pipeline-based speech translation [(Baltic HLT '20)](https://hal.inria.fr/hal-02907053) | WMT '17 | 
| Lexical-Constraint-Aware Neural Machine Translation via Data Augmentation [(IJCAI '20)](https://www.ijcai.org/proceedings/2020/496) [code](https://github.com/ghchen18/leca) | WMT '16 de-en, NIST zh-en |
| A Diverse Data Augmentation Strategy for Low-Resource Neural Machine Translation [(Information '20)](https://www.mdpi.com/2078-2489/11/5/255) | IWSLT '14 en-de | 
| Syntax-aware Data Augmentation for Neural Machine Translation [(arxiv '20)](https://arxiv.org/abs/2004.14200) | WMT '14 en-de, IWSLT '14 de-en | 
| SSMBA: Self-Supervised Manifold Based Data Augmentation for Improving Out-of-Domain Robustness [(EMNLP '20)](https://www.aclweb.org/anthology/2020.emnlp-main.97/) [code](https://github.com/nng555/ssmba) | IWSLT'14 | 
| Data diversification: A simple strategy for neural machine translation [(NeurIPS '20)](https://proceedings.neurips.cc/paper/2020/file/7221e5c8ec6b08ef6d3f9ff3ce6eb1d1-Paper.pdf) [code](https://github.com/nxphi47/data_diversification) | WMT '14 en-de/en-fr, IWSLT '13/'14/'15 en-de/de-en/en-fr |
| AdvAug: Robust Adversarial Augmentation for Neural Machine Translation [(ACL '20)](https://www.aclweb.org/anthology/2020.acl-main.529/) | NIST zh-en, WMT '14 en-de| 
| Dictionary-based Data Augmentation for Cross-Domain Neural Machine Translation [(arxiv '20)](https://arxiv.org/abs/2004.02577) | WMT '14/'19 | 
| Sentence Boundary Augmentation For Neural Machine Translation Robustness [(arxiv '20)](https://arxiv.org/abs/2010.11132) | IWSLT '14/'15/'18 en-de, WMT '18 en-de | 
| Valar nmt : Vastly lacking resources neural machine translation [(Stanford CS224N)](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/custom/15811193.pdf) | Bible, Misc, Europarl v8, Newstest '18 | 


### Summarization

| Paper | Datasets | 
| -- | --- |
| Transforming Wikipedia into Augmented Data for Query-Focused Summarization [(arxiv '19)](https://arxiv.org/abs/1911.03324) | DUC |
| Iterative Data Augmentation with Synthetic Data (Abstract Text Summarization: A Low Resource Challenge [(EMNLP '19)](https://www.aclweb.org/anthology/D19-1616/) | Swisstext, commoncrawl | 
| Improving Zero and Few-Shot Abstractive Summarization with Intermediate Fine-tuning and Data Augmentation [(NAACL '21)](https://arxiv.org/abs/2010.12836) | CNN-DailyMail | 
| Data Augmentation for Abstractive Query-Focused Multi-Document Summarization [(AAAI '21)](https://arxiv.org/abs/2103.01863) [code](https://github.com/ramakanth-pasunuru/QmdsCnnIr) | QMDSCNN, QMDSIR, WikiSum, DUC 2006, DUC 2007 |


### Question Answering

| Paper | Datasets | 
| -- | --- |
| QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension [(ICLR '18)](https://openreview.net/forum?id=B14TlG-RW) | SQuAD, TriviaQA |
| An Exploration of Data Augmentation and Sampling Techniques for Domain-Agnostic Question Answering [(EMNLP '19 Workshop)](https://www.aclweb.org/anthology/D19-5829/) | MRQA | 
| Data Augmentation for BERT Fine-Tuning in Open-Domain Question Answering [(arxiv '19)](https://arxiv.org/abs/1904.06652) | SQuAD, Trivia-QA, CMRC, DRCD | 
| XLDA: Cross-Lingual Data Augmentation for Natural Language Inference and Question Answering [(arxiv '19)](https://openreview.net/forum?id=BJgAf6Etwr) | XNLI, SQuAD |
| Synthetic Data Augmentation for Zero-Shot Cross-Lingual Question Answering [(arxiv '20)](https://arxiv.org/abs/2010.12643) | MLQA, XQuAD, SQuAD-it, PIAF | 
| Logic-Guided Data Augmentation and Regularization for Consistent Question Answering [(ACL '20)](https://www.aclweb.org/anthology/2020.acl-main.499/) [code](https://github.com/AkariAsai/logic_guided_qa) | WIQA, QuaRel, HotpotQA |


### Sequence Tagging

| Paper | Datasets | 
| -- | --- |
| Data Augmentation via Dependency Tree Morphing for Low-Resource Languages [(EMNLP '18)](https://www.aclweb.org/anthology/D18-1545.pdf) [code](https://github.com/gozdesahin/crop-rotate-augment) | universal dependencies project | 
| DAGA: Data Augmentation with a Generation Approach for Low-resource Tagging Tasks [(EMNLP '20)](https://www.aclweb.org/anthology/2020.emnlp-main.488/) [code](https://github.com/ntunlp/daga) | CoNLL2002/2003 |
| An Analysis of Simple Data Augmentation for Named Entity Recognition [(COLING '20)](https://www.aclweb.org/anthology/2020.coling-main.343/) | MaSciP, i2b2- 2010 |
| SeqMix: Augmenting Active Sequence Labeling via Sequence Mixup [(EMNLP '20)](https://www.aclweb.org/anthology/2020.emnlp-main.691/) [code](https://github.com/rz-zhang/SeqMix) | CoNLL-03, ACE05, Webpage |


### Parsing
| Paper | Datasets | 
| -- | --- |
| Data Recombination for Neural Semantic Parsing [(ACL '16)](https://www.aclweb.org/anthology/P16-1002/) [code](https://github.com/dongpobeyond/Seq2Act) | GeoQuery, ATIS, Overnight |
| A systematic comparison of methods for low-resource dependency parsing on genuinely low-resource languages [(EMNLP '19)](https://www.aclweb.org/anthology/D19-1102/) | Universal Dependencies treebanks version 2.2 |
| Named Entity Recognition for Social Media Texts with Semantic Augmentation [(EMNLP '20)](https://www.aclweb.org/anthology/2020.emnlp-main.107/)[code](https://github.com/cuhksz-nlp/SANER) | WNUT16, WNUT17, Weibo |
| Good-Enough Compositional Data Augmentation [(ACL '20)](https://www.aclweb.org/anthology/2020.acl-main.676/) [code](https://github.com/jacobandreas/geca) | SCAN |
| GraPPa: Grammar-Augmented Pre-Training for Table Semantic Parsing [(ICLR '21)](https://openreview.net/forum?id=kyaIeYj4zZ) | SPIDER, WIKISQL, WIKITABLEQUESTIONS |


### Grammatical Error Correction
| Paper | Datasets | 
| -- | --- |
| GenERRate: Generating Errors for Use in Grammatical Error Detection [(BEA '09)](https://www.aclweb.org/anthology/W09-2112/) | Ungram-BNC |
| Mining Revision Log of Language Learning SNS for Automated Japanese Error Correction of Second Language Learners [(IJCNLP '11)](https://www.aclweb.org/anthology/I11-1017/) [code](https://github.com/google-research-datasets/clang8) | Lang-8 |
| Artificial error generation for translation-based grammatical error correction [(University of Cambridge Technical Report '16)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-895.pdf)  | Several Datasets |
| Noising and Denoising Natural Language: Diverse Backtranslation for Grammar Correction. [(NAACL'18)](https://www.aclweb.org/anthology/N18-1057/) | Lang-8, CoNLL-2014, CoNLL-2013, JFLEG | 
| Using Wikipedia Edits in Low Resource Grammatical Error Correction. [(WNUT @ EMNLP '18)](https://doi.org/10.18653/v1/W18-6111) | Falko-MERLIN GEC Corpus |
| Sequence-to-sequence Pre-training with Data Augmentation for Sentence Rewriting [(arxiv '19)](https://arxiv.org/abs/1909.06002) | CoNLL-2014 , JFLEG |
| Controllable Data Synthesis Method for Grammatical Error Correction [(arxiv '19)](https://arxiv.org/abs/1909.13302) [code](https://github.com/marumalo/survey/issues/21) | NUCLE, Lang-8, One-Billion, CoNLL2013, CoNLL2014|
| Neural Grammatical Error Correction Systems with Unsupervised Pre-training on Synthetic Data. [(BEA @ ACL '19)](https://doi.org/10.18653/v1/W19-4427) | FCE, NUCLE, W&I+LOCNESS, Lang-8 |
| Corpora Generation for Grammatical Error Correction [(NAACL'19)](https://doi.org/10.18653/v1/N19-1333) | CoNLL-2014, JFLEG, Lang-8 |
| Erroneous data generation for Grammatical Error Correction [(BEA @ ACL '19)](https://www.aclweb.org/anthology/W19-4415/) | Lang-8,n CoNLL, JFLEG, CoNLL-2014, ABCN, FCE |
| Sequence-to-sequence Pre-training with Data Augmentation for Sentence Rewriting [(arxiv '19)](https://arxiv.org/abs/1909.06002) [code](https://github.com/marumalo/survey/issues/6) | GYAFC, WMT14, WMT18 |
| A neural grammatical error correction  system  built  on  better  pre-training  and  sequential  transfer  learning. [(BEA @ ACL '19)](https://doi.org/10.18653/v1/W19-4423) | FCE, NUCLE, W&I+LOCNESS, Lang-8, Gutenberg, Tatoeba, WikiText-103 |
| Improving Grammatical Error Correction with Data Augmentation by Editing Latent Representation [(COLING'20)](https://doi.org/10.18653/v1/2020.coling-main.200) | FCE, NUCLE, W&I+LOCNESS, Lang-8 |
| A Comparative Study of Synthetic Data Generation Methods for Grammatical Error Correction [(BEA @ ACL '20)](https://www.aclweb.org/anthology/2020.bea-1.21/) | W&I+LOCNESS, FCE, News Crawl 2, W&I+L train, FCE-train, NUCLE, Lang-8, W&I+L dev, FCE-test, Tatoeba, WikiText-103 |
| A syntactic rule-based framework for parallel data synthesis in Japanese GEC [(MIT Thesis '20)](https://dspace.mit.edu/handle/1721.1/127416) | Lang-8 |


### Generation

| Paper | Datasets | 
| -- | --- |
| TNT-NLG, System 2: Data repetition and meaning representation manipulation to improve neural generation [(E2E NLG Challenge System Descriptions)](http://www.macs.hw.ac.uk/InteractionLab/E2E/final_papers/E2E-TNT_NLG2.pdf) | TODO | 
| Findings of the Third Workshop on Neural Generation and Translation [(WNGT @ EMNLP '19)](https://www.aclweb.org/anthology/D19-5601/) | RotoWire English-German | 
| A Good Sample is Hard to Find: Noise Injection Sampling and Self-Training for Neural Language Generation Models [(INLG '19)](https://www.aclweb.org/anthology/W19-8672/) [code](https://github.com/kedz/noiseylg) | E2E Challenge Dataset, Laptops, TVs | 
| GenAug: Data Augmentation for Finetuning Text Generators [(DeeLIO @ EMNLP '20)](https://www.aclweb.org/anthology/2020.deelio-1.4/) [code](https://github.com/styfeng/GenAug) | Yelp | 
| Denoising Pre-Training and Data Augmentation Strategies for Enhanced RDF Verbalization with Transformers [(WebNLG+ @ INLG '20)](https://www.aclweb.org/anthology/2020.webnlg-1.9/) | WebNLG |


### Dialogue
| Paper | Datasets | 
| -- | --- |
| Sequence-to-Sequence Data Augmentation for Dialogue Language Understanding [(COLING '18)](https://www.aclweb.org/anthology/C18-1105/) [code](https://github.com/AtmaHou/Seq2SeqDataAugmentationForLU) | ATIS, Dec94, Stanford dialogue |
| Task-Oriented Dialog Systems that Consider Multiple Appropriate Responses under the Same Context [(arxiv '19)](https://arxiv.org/abs/1911.10484) [code](https://github.com/thu-spmi/damd-multiwoz) | MultiWOZ |
| Data Augmentation by Data Noising for Open-vocabulary Slots in Spoken Language Understanding [(Student Research Workshop @ NAACL '19)](https://www.aclweb.org/anthology/N19-3014/) | ATIS, Snips, MR |
| Data Augmentation with Atomic Templates for Spoken Language Understanding [(EMNLP '19)](https://www.aclweb.org/anthology/D19-1375/) [code](https://github.com/sz128/DAAT_SLU) | DSTC 2&3,  DSTC2 |
| Data Augmentation for Spoken Language Understanding via Joint Variational Generation [(AAAI '19)](https://ojs.aaai.org/index.php/AAAI/article/view/4729) | ATIS, Snips, MIT |
| Effective Data Augmentation Approaches to End-to-End Task-Oriented Dialogue [(IALP '19)](https://ieeexplore.ieee.org/document/9037690) | CamRest676, KVRET |
| Paraphrase Augmented Task-Oriented Dialog Generation [(ACL '20)](https://www.aclweb.org/anthology/2020.acl-main.60/) [code](https://github.com/thu-spmi/PARG) | TCamRest676, MultiWOZ |
| Dialog State Tracking with Reinforced Data Augmentation [(AAAI '20)](https://ojs.aaai.org/index.php/AAAI/article/view/6491) | WoZ,  MultiWoZ |
| Data Augmentation for Copy-Mechanism in Dialogue State Tracking [(arxiv '20)](https://arxiv.org/abs/2002.09634) | WoZ, DSTC2, Multi |
| Simple is Better! Lightweight Data Augmentation for Low Resource Slot Filling and Intent Classification [(PACLIC '20)](https://www.aclweb.org/anthology/2020.paclic-1.20/) [code](https://github.com/slouvan/saug) | ATIS, SNIPS, FB |
| Conversation Graph: Data Augmentation, Training, and Evaluation for Non-Deterministic Dialogue Management [(TACL '21)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00352/97777/Conversation-Graph-Data-Augmentation-Training-and) | M2M, MultiWOZ |

### Multimodal
| Paper | Datasets | 
| -- | --- |
| Data Augmentation for Visual Question Answering [(INLG '17)](https://www.aclweb.org/anthology/W17-3529/) | COCO-VQA, COCO-QA |
| Low Resource Multi-modal Data Augmentation for End-to-end ASR [(CoRR ’18)](https://deepai.org/publication/low-resource-multi-modal-data-augmentation-for-end-to-end-asr) | TODO |
| Multi-Modal Data Augmentation for End-to-end ASR [(Interspeech '18)](https://www.isca-speech.org/archive/Interspeech_2018/abstracts/2456.html) | Voxforge, HUB4 |
| Augmenting Image Question Answering Dataset by Exploiting Image Captions [(LREC '18)](https://www.aclweb.org/anthology/L18-1436/) | IQA |
| Multimodal Continuous Emotion Recognition with Data Augmentation Using Recurrent Neural Networks [(AVEC '18)](https://dl.acm.org/doi/10.1145/3266302.3266304) | TODO |
| Multimodal Dialogue State Tracking By QA Approach with Data Augmentation [(DSTC8 @ AAAI '20)](https://arxiv.org/abs/2007.09903) | DSTC7-AVSD |
| Data augmentation techniques for the Video Question Answering task [(arxiv '20)](https://arxiv.org/abs/2008.09849) | TGIF-QA,  MSVD-QA |
| Data Augmentation for Training Dialog Models Robust to Speech Recognition Errors [(NLP for ConvAI @ ACL '20)](https://arxiv.org/abs/2006.05635) | DSTC2 |
| Semantic Equivalent Adversarial Data Augmentation for Visual Question Answering [(ECCV '20)](https://link.springer.com/chapter/10.1007/978-3-030-58529-7_26) | TODO |
| Text Augmentation Using BERT for Image Captioning [(Applied Sciences '20)](https://www.mdpi.com/2076-3417/10/17/5978) | MSCOCO |
| MDA: Multimodal Data Augmentation Framework for Boosting Performance on Image-Text Sentiment/Emotion Classification Tasks [(IEEE Intelligent Systems '20)](https://ieeexplore.ieee.org/document/9206007) | TODO |

### Mitigating Bias
| Paper | Datasets | 
| -- | --- |
| Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods. [(NAACL '18)](https://www.aclweb.org/anthology/N18-2003/) [code](https://github.com/uclanlp/corefBias) | WinoBias, OntoNotes|
| Counterfactual Data Augmentation for Mitigating Gender Stereotypes in Languages with Rich Morphology [(ACL '19)](https://www.aclweb.org/anthology/P19-1161/) [code](https://github.com/rycolab/biasCDA) | TODO |
| It’s All in the Name: Mitigating Gender Bias with Name-Based Counterfactual Data Substitution [(EMNLP '19)](https://www.aclweb.org/anthology/D19-1530/) [code](https://github.com/rowanhm/counterfactual-data-substitution) | SSA, Stanford Large Movie Review, SimLex-999 |
| Gender Bias in Neural Natural Language Processing. [(Springer '20)](https://link.springer.com/chapter/10.1007%2F978-3-030-62077-6_14 ) | Wikitext-2, CoNLL-2012 |
| Improving Robustness by Augmenting Training Sentences with Predicate-Argument Structures [(arxiv '20)](https://arxiv.org/abs/2010.12510) | SWAG, CoNLL2009, MultiNLI, HANS|

### Mitigating Class Imbalance
| Paper | Datasets | 
| -- | --- |
| SMOTE: Synthetic Minority Over-sampling Technique [(Journal of Artificial Intelligence Research '02)](https://www.jair.org/index.php/jair/article/view/10302) | Pima, Phoneme, Adult, E-state, Satimage, Forest Cover, Oil, Mammography, Can |
| Active Learning for Word Sense Disambiguation with Methods for Addressing the Class Imbalance Problem [(EMNLP '07)](https://www.aclweb.org/anthology/D07-1082/) | TODO |
| MLSMOTE: Approaching imbalanced multilabel learning through synthetic instance generation [(Knowledge-Based Systems '15)](https://www.sciencedirect.com/science/article/abs/pii/S0950705115002737?via%3Dihub) | bibtex, cal500, corel5k, slashdot, tmc2007, mediamill, medical, scene, enron, emotions |
| SMOTE for Learning from Imbalanced Data: Progress and Challenges, Marking the 15-year Anniversary [(Journal of Artificial Intelligence Research '18)](https://www.jair.org/index.php/jair/article/view/11192) | TODO |

### Adversarial examples

| Paper | Datsets | 
| -- | --- |
| Adversarial Example Generation with Syntactically Controlled Paraphrase Networks [(NAACL '18)](https://www.aclweb.org/anthology/N18-1170/) | SST, SICK | 
| Certified Robustness to Adversarial Word Substitutions [(EMNLP '19)](https://www.aclweb.org/anthology/D19-1423/) | TODO | 
| PAWS: Paraphrase Adversaries from Word Scrambling [(NAACL '19)](https://www.aclweb.org/anthology/N19-1131/) | TODO | 
| AdvEntuRe: Adversarial Training for Textual Entailment with Knowledge-Guided Examples [(ACL '18)](https://www.aclweb.org/anthology/P18-1225/) | TODO | 
| Breaking NLI Systems with Sentences that Require Simple Lexical Inferences [(ACL '18)](https://www.aclweb.org/anthology/P18-2103/) | TODO |

### Compositionality

| Paper | Datsets | 
| -- | --- |
| Good-Enough Compositional Data Augmentation [(ACL '20)](https://www.aclweb.org/anthology/2020.acl-main.676.pdf) [code](https://github.com/jacobandreas/geca) | SCAN |
| Sequence-Level Mixed Sample Data Augmentation [(EMNLP '20)](https://www.aclweb.org/anthology/2020.emnlp-main.447) [code](https://github.com/dguo98/seqmix) | IWSLT ’14, WMT ’14 | 

### Automated Augmentation

| Paper                                                        | Datsets                     |
| ------------------------------------------------------------ | --------------------------- |
| Learning Data Manipulation for Augmentation and Weighting [(NeurIPS '19)](https://papers.nips.cc/paper/2019/file/671f0311e2754fcdd37f70a8550379bc-Paper.pdf) [code](https://github.com/tanyuqian/learning-data-manipulation) | SST, IMDB, TREC, CIFAR-10   |
| Data Manipulation: Towards Effective Instance Learning for Neural Dialogue Generation via Learning to Augment and Reweight [(ACL '20)](https://www.aclweb.org/anthology/2020.acl-main.564.pdf) | DailyDialog,  OpenSubtitles |


### Popular Resources
- [A visual survey of data augmentation in NLP](https://amitness.com/2020/05/data-augmentation-for-nlp/)
- [nlpaug](https://github.com/makcedward/nlpaug)
- [TextAttack](https://github.com/QData/TextAttack)
- [AugLy](https://github.com/facebookresearch/AugLy)
