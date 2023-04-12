# UniHash
Pytorch implementation of the paper "Contrastive Label Correlation Enhanced Unified Hashing Encoder for Cross-modal Retrieval", CIKM'22 
 paper. Please remember to give a citation if [UniHash](https://dl.acm.org/doi/abs/10.1145/3511808.3557265) and codes benefits your research!
```
@inproceedings{wu2022contrastive,
  title={Contrastive Label Correlation Enhanced Unified Hashing Encoder for Cross-modal Retrieval},
  author={Wu, Hongfa and Zhang, Lisai and Chen, Qingcai and Deng, Yimeng and Siebert, Joanna and Han, Yunpeng and Li, Zhonghua and Kong, Dejiang and Cao, Zhao},
  booktitle={Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  pages={2158--2168},
  year={2022}
}
```
# Environment Setup
Install the python packages following [VinVL](https://github.com/microsoft/Oscar/blob/master/INSTALL.md). 

# Dataset and Pre-processed Files
You need to download the dataset NUS-WIDE, IAPR-TC, MIRFlickr-25k to reproduce the exprements. 
Then you need [pre-processed label files](https://drive.google.com/file/d/1w7HuIqffju7joFqUWKiqCwXd3QNlna75/view?usp=share_link) from google drive, and save them into corresponding datasets.

Then, please refer to [Scene Graph](https://github.com/microsoft/scene_graph_benchmark) to extract the image features. 
 

## Initialize weights (optional)
The decomposition is applied on the pre-trained one stream VinVL [model](https://github.com/microsoft/Oscar/blob/master/VinVL_DOWNLOAD.md#pre-trained-models), so you need to download it first.
```
path/to/azcopy copy 'https://biglmdiag.blob.core.windows.net/vinvl/model_ckpts/TASK_NAME' coco_ir --recursive
```

Afterwards, you can run our code to train UniHash.


## Pre-trained Model Checkpoints
You can also directly use our pre-trained checkpoints for [IAPR](https://drive.google.com/file/d/1h4nNR8_hORjtwZpOvf4K2Z5HG1J2q3nr/view?usp=share_link).

# Running
Run contrastive_learn.py using following args (take NUS-WIDE as an example):
```
"program": "${workspaceFolder}/try_contrastive_new.py",
"args": [
    "--tagslabel", "NUS-WIDE/img_tagslabel.json", # from proided files
    "--class_name", "NUS-WIDE/class_name.json", # from proided files
    "--img_feat_file", "NUS-WIDE/vinvl_vg_x152c4/predictions.tsv", # extracted by Scene Graph
    "--eval_model_dir", "vinvl/checkpoint-234-15000", # initialized weights
    "--do_lower_case",
    "--output_dir",
    "output/coco_base",
    "--training_size","10000",
    "--query_size","5000",
    "--database_size","20000",
    "--class_number","80",
    "--bit_num","64",
    "--batch_size","128",
    "--train",
    "--test",
]
       
```
For test, you can use the following args:
```
"program": "${workspaceFolder}/base.py",
"args": [
    "--tagslabel", "NUS-WIDE/img_tagslabel.json", 
    "--class_name", "NUS-WIDE/class_name.json",
    "--img_feat_file", "NUS-WIDE/vinvl_vg_x152c4/features.tsv",
    "--eval_model_dir", "/data/checkpoint/checkpoint-1340000/",
    "--do_lower_case",
    "--output_dir",
    "output/coco_base",
    "--training_size","10000",
    "--query_size","5000",
    "--database_size","20000",
    "--class_number","80",
    "--bit_num","64",
    "--batch_size","128",
    "--test",
]
```


# Acknowledge
This repo is modified based on the [VinVL](https://github.com/microsoft/Oscar), we thank the authors for sharing their project.
