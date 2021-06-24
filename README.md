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

### Natural Language Generation

| Paper | Datasets | 
| -- | --- |
| GenAug: Data Augmentation for Finetuning Text Generators [(DeeLIO @ EMNLP '20)](https://www.aclweb.org/anthology/2020.deelio-1.4/) [code](https://github.com/styfeng/GenAug) | Yelp | 
| Findings of the Third Workshop on Neural Generation and Translation [(WNGT @ EMNLP '19)](https://www.aclweb.org/anthology/D19-5601/) | TODO | 
| Denoising Pre-Training and Data Augmentation Strategies for Enhanced RDF Verbalization with Transformers [(WebNLG+ @ INLG '20)](https://www.aclweb.org/anthology/2020.webnlg-1.9/) | TODO | 
| TNT-NLG, System 2: Data repetition and meaning representation manipulation to improve neural generation [(E2E NLG Challenge System Descriptions)](http://www.macs.hw.ac.uk/InteractionLab/E2E/final_papers/E2E-TNT_NLG2.pdf) | TODO | 
| A Good Sample is Hard to Find: Noise Injection Sampling and Self-Training for Neural Language Generation Models [(INLG '19)](https://www.aclweb.org/anthology/W19-8672/) | TODO | 


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
| QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension [(ICLR '18)](https://openreview.net/forum?id=B14TlG-RW) | TODO |

### Summarization

| Paper | Datasets | 
| -- | --- |
| Transforming Wikipedia into Augmented Data for Query-Focused Summarization [(arxiv '19)](https://arxiv.org/abs/1911.03324) | DUC |
| Iterative Data Augmentation with Synthetic Data (Abstract Text Summarization: A Low Resource Challenge [(EMNLP '19)](https://www.aclweb.org/anthology/D19-1616/) | Swisstext, commoncrawl | 
| Improving Zero and Few-Shot Abstractive Summarization with Intermediate Fine-tuning and Data Augmentation [(NAACL '21)](https://arxiv.org/abs/2010.12836) | CNN-DailyMail | 
| Data Augmentation for Abstractive Query-Focused Multi-Document Summarization [(AAAI '21)](https://arxiv.org/abs/2103.01863) | TODO | 


### Sequence Tagging

| Paper | Datasets | 
| -- | --- |
| Data Augmentation via Dependency Tree Morphing for Low-Resource Languages [(EMNLP '18)](https://www.aclweb.org/anthology/D18-1545.pdf) [code](https://github.com/gozdesahin/crop-rotate-augment) | universal dependencies project | 
| DAGA: Data Augmentation with a Generation Approach for Low-resource Tagging Tasks [(EMNLP '20)](https://www.aclweb.org/anthology/2020.emnlp-main.488/) | TODO |
| An Analysis of Simple Data Augmentation for Named Entity Recognition [(COLING '20)](https://www.aclweb.org/anthology/2020.coling-main.343/) | TODO |
| SeqMix: Augmenting Active Sequence Labeling via Sequence Mixup [(EMNLP '20)](https://www.aclweb.org/anthology/2020.emnlp-main.691/) | TODO |

### Parsing
| Paper | Datasets | 
| -- | --- |
| Named Entity Recognition for Social Media Texts with Semantic Augmentation [(EMNLP '20)](https://www.aclweb.org/anthology/2020.emnlp-main.107/) | TODO |
| Data Recombination for Neural Semantic Parsing [(ACL '16)](https://www.aclweb.org/anthology/P16-1002/) | TODO |
| GraPPa: Grammar-Augmented Pre-Training for Table Semantic Parsing [(ICLR '21)](https://openreview.net/forum?id=kyaIeYj4zZ) | TODO |
| Good-Enough Compositional Data Augmentation [(ACL '20)](https://www.aclweb.org/anthology/2020.acl-main.676/) | TODO |
| A systematic comparison of methods for low-resource dependency parsing on genuinely low-resource languages [(EMNLP '19)](https://www.aclweb.org/anthology/D19-1102/) | TODO |


### Grammatical Error Correction
| Paper | Datasets | 
| -- | --- |
| Using  Wikipedia  Edits  in  Low Resource Grammatical Error Correction. [(WNUT @ EMNLP '18)](https://doi.org/10.18653/v1/W18-6111) | Falko-MERLIN GEC Corpus |
| Sequence-to-sequence Pre-training with Data Augmentation for Sentence Rewriting [(arxiv '19)](https://arxiv.org/abs/1909.06002) | CoNLL-2014 , JFLEG  |
| Controllable Data Synthesis Method for Grammatical Error Correction [(arxiv '19)](https://arxiv.org/abs/1909.13302) | TODO |
| Neural Grammatical Error Correction  Systems  with  Unsupervised  Pre-training on Synthetic Data. [(BEA @ ACL '19)](https://doi.org/10.18653/v1/W19-4427) | FCE, NUCLE, W&I+LOCNESS, Lang-8 (BEA @ ACL '19 Shared Task) |
| A neural grammatical error cor-rection  system  built  on  better  pre-training  and  se-quential  transfer  learning. [(BEA @ ACL '19)](https://doi.org/10.18653/v1/W19-4423) | FCE, NUCLE, W&I+LOCNESS, Lang-8 (BEA @ ACL '19 Shared Task), Gutenberg, Tatoeba, WikiText-103 (Pretraining) |
| Improving  Grammatical  Error  Correction with  Data  Augmentation  by  Editing  Latent  Representation [(COLING'20)](https://doi.org/10.18653/v1/2020.coling-main.200) | FCE, NUCLE, W&I+LOCNESS, Lang-8 (BEA @ ACL '19 Shared Task)  |
| Noising and Denoising Natural Language:  Diverse Backtranslation for Grammar  Correction. [(NAACL'18)](https://www.aclweb.org/anthology/N18-1057/)  | Lang-8, CoNLL-2014, CoNLL-2013, JFLEG |
| Corpora Generation for Grammatical Error Correction [(NAACL'19)](https://doi.org/10.18653/v1/N19-1333)  | CoNLL-2014, JFLEG, Lang-8 |
| A Comparative Study of Synthetic Data Generation Methods for Grammatical Error Correction [(BEA @ ACL '20)](https://www.aclweb.org/anthology/2020.bea-1.21/)  | TODO |
| GenERRate: Generating Errors for Use in Grammatical Error Detection [(BEA '09)](https://www.aclweb.org/anthology/W09-2112/)  | TODO |
| A syntactic rule-based framework for parallel data synthesis in Japanese GEC [(MIT Thesis '20)](https://dspace.mit.edu/handle/1721.1/127416)  | TODO |
| Artificial error generation for translation-based grammatical error correction [(University of Cambridge Technical Report)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-895.pdf)  | TODO |
| Erroneous data generation for Grammatical Error Correction [(BEA @ ACL '19)](https://www.aclweb.org/anthology/W19-4415/)  | TODO |
| Sequence-to-sequence Pre-training with Data Augmentation for Sentence Rewriting [(arxiv '19)](https://arxiv.org/abs/1909.06002)  | TODO |
| Mining Revision Log of Language Learning SNS for Automated Japanese Error Correction of Second Language Learners [(IJCNLP '11)](https://www.aclweb.org/anthology/I11-1017/)  | TODO |

### Dialogue
| Paper | Datasets | 
| -- | --- |
| Effective Data Augmentation Approaches to End-to-End Task-Oriented Dialogue [(IALP '19)](https://ieeexplore.ieee.org/document/9037690) | TODO |
| Simple is Better! Lightweight Data Augmentation for Low Resource Slot Filling and Intent Classification [(PACLIC '20)](https://www.aclweb.org/anthology/2020.paclic-1.20/) | TODO |
| Sequence-to-Sequence Data Augmentation for Dialogue Language Understanding [(COLING '18)](https://www.aclweb.org/anthology/C18-1105/) | TODO |
| Task-Oriented Dialog Systems that Consider Multiple Appropriate Responses under the Same Context [(arxiv '19)](https://arxiv.org/abs/1911.10484) | TODO |
| Data Augmentation by Data Noising for Open-vocabulary Slots in Spoken Language Understanding [(Student Research Workshop @ NAACL '19)](https://www.aclweb.org/anthology/N19-3014/) | TODO |
| Data Augmentation with Atomic Templates for Spoken Language Understanding [(EMNLP '19)](https://www.aclweb.org/anthology/D19-1375/) | TODO |
| Data Augmentation for Spoken Language Understanding via Joint Variational Generation [(AAAI '19)](https://ojs.aaai.org/index.php/AAAI/article/view/4729) | TODO |
| Paraphrase Augmented Task-Oriented Dialog Generation [(ACL '20)](https://www.aclweb.org/anthology/2020.acl-main.60/) | TODO |
| Conversation Graph: Data Augmentation, Training, and Evaluation for Non-Deterministic Dialogue Management [(TACL '21)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00352/97777/Conversation-Graph-Data-Augmentation-Training-and) | TODO |
| Dialog State Tracking with Reinforced Data Augmentation [(AAAI '20)](https://ojs.aaai.org/index.php/AAAI/article/view/6491) | TODO |
| Data Augmentation for Copy-Mechanism in Dialogue State Tracking [(arxiv '20)](https://arxiv.org/abs/2002.09634) | TODO |

### Multimodal
| Paper | Datasets | 
| -- | --- |
| Data Augmentation for Training Dialog Models Robust to Speech Recognition Errors [(NLP for ConvAI @ ACL '20)](https://arxiv.org/abs/2006.05635) | TODO |
| Low Resource Multi-modal Data Augmentation for End-to-end ASR [(CoRR)](https://deepai.org/publication/low-resource-multi-modal-data-augmentation-for-end-to-end-asr) | TODO |
| Multi-Modal Data Augmentation for End-to-end ASR [(Interspeech '18)](https://www.isca-speech.org/archive/Interspeech_2018/abstracts/2456.html) | TODO |
| MDA: Multimodal Data Augmentation Framework for Boosting Performance on Image-Text Sentiment/Emotion Classification Tasks [(IEEE Intelligent Systems '20)](https://ieeexplore.ieee.org/document/9206007) | TODO |
| Text Augmentation Using BERT for Image Captioning [(Applied Sciences '20)](https://www.mdpi.com/2076-3417/10/17/5978) | TODO |
| Data Augmentation for Visual Question Answering [(INLG '17)](https://www.aclweb.org/anthology/W17-3529/) | TODO |
| Augmenting Image Question Answering Dataset by Exploiting Image Captions [(LREC '18)](https://www.aclweb.org/anthology/L18-1436/) | TODO |
| Semantic Equivalent Adversarial Data Augmentation for Visual Question Answering [(ECCV '20)](https://link.springer.com/chapter/10.1007/978-3-030-58529-7_26) | TODO |
| Multimodal Continuous Emotion Recognition with Data Augmentation Using Recurrent Neural Networks [(AVEC '18)](https://dl.acm.org/doi/10.1145/3266302.3266304) | TODO |
| Multimodal Dialogue State Tracking By QA Approach with Data Augmentation [(DSTC8 @ AAAI '20)](https://arxiv.org/abs/2007.09903) | TODO |
| Data augmentation techniques for the Video Question Answering task [(arxiv '20)](https://arxiv.org/abs/2008.09849) | TODO |

### Mitigating Bias
| Paper | Datasets | 
| -- | --- |
| Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods. [(NAACL '18)](https://www.aclweb.org/anthology/N18-2003/) | TODO |
| Gender Bias in Neural Natural Language Processing. [(Springer '20)](https://link.springer.com/chapter/10.1007%2F978-3-030-62077-6_14) | TODO |
| Counterfactual Data Augmentation for Mitigating Gender Stereotypes in Languages with Rich Morphology [(ACL '19)](https://www.aclweb.org/anthology/P19-1161/) | TODO |
| It’s All in the Name: Mitigating Gender Bias with Name-Based Counterfactual Data Substitution [(EMNLP '19)](https://www.aclweb.org/anthology/D19-1530/) | TODO |
| Improving Robustness by Augmenting Training Sentences with Predicate-Argument Structures [(arxiv '20)](https://arxiv.org/abs/2010.12510) | TODO |

### Mitigating Class Imbalance
| Paper | Datasets | 
| -- | --- |
| SMOTE: Synthetic Minority Over-sampling Technique [(Journal of Artificial Intelligence Research '02)](https://www.jair.org/index.php/jair/article/view/10302) | TODO |
| SMOTE for Learning from Imbalanced Data: Progress and Challenges, Marking the 15-year Anniversary [(Journal of Artificial Intelligence Research '18)](https://www.jair.org/index.php/jair/article/view/11192) | TODO |
| MLSMOTE: Approaching imbalanced multilabel learning through synthetic instance generation [(Knowledge-Based Systems '15)](https://www.sciencedirect.com/science/article/abs/pii/S0950705115002737?via%3Dihub) | TODO |
| Active Learning for Word Sense Disambiguation with Methods for Addressing the Class Imbalance Problem [(EMNLP '07)](https://www.aclweb.org/anthology/D07-1082/) | TODO |

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
| Sequence-Level Mixed Sample Data Augmentation [(EMNLP '20)](https://www.aclweb.org/anthology/2020.emnlp-main.447) [code](https://github.com/dguo98/seqmix) | SCAN | 

### Popular Resources
- [A visual survey of data augmentation in NLP](https://amitness.com/2020/05/data-augmentation-for-nlp/)
- [nlpaug](https://github.com/makcedward/nlpaug)
- [TextAttack](https://github.com/QData/TextAttack)
