# Cross-lingual ABSA (XABSA)
This repo contains the data and code for our paper [Cross-lingual Aspect-based Sentiment Analysis with Aspect Term Code-Switching ](https://aclanthology.org/2021.emnlp-main.727.pdf) in EMNLP 2021.


## Requirements
- torch==1.3.1
- numpy==1.19.4
- transformers==3.4.0 (You can also use the static version in this repo)
- sentencepiece==0.1.91
- tokenizer==0.9.2
- sacremoses==0.0.43


##  Quick Start
- Download the pre-trained multilingual language model mBERT or XLM-R
- To quickly reproduce the results with French (`fr`) as the target langauge and mBERT (`mbert`) as the backbone under the supervised setting:
```python
python main.py --tfm_type mbert --tgt_lang fr
```
- To reproduce other results, ref to the `data_utils.py` for details


## Usage
To run experiments under different settings, change the `exp_type` setting:
  * `supervised` refers to the supervised setting
  * `acs` is the proposed method
  * `acs_kd_s/m`: single/multi teacher distillation under the cross-lingual setting
  * `macs_kd`: multilingual distillation

Two example scripts:
- `run_absa.sh` provides an example to run basic experiment.
- `run_absa_kd.sh` provides an example to run experiments with knowledge distilling on the unlabeled target laguage data.


## Citation
If the code is used in your research, please star our repo and cite our paper as follows:
```
@inproceedings{zhang-etal-2021-cross,
    title = "Cross-lingual Aspect-based Sentiment Analysis with Aspect Term Code-Switching",
    author = "Zhang, Wenxuan  and
      He, Ruidan  and
      Peng, Haiyun  and
      Bing, Lidong  and
      Lam, Wai",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.727",
    pages = "9220--9230",
}
```
​