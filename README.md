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

Moreover, we use the features extracted by VinVL, which are given in their [download page](https://github.com/microsoft/Oscar/blob/master/VinVL_DOWNLOAD.md).


If you want to run the model on your customed data, please refer to [Scene Graph](https://github.com/microsoft/scene_graph_benchmark) to extract the features, which is specified by the VinVL repo.
 

# Pre-trained Model Checkpoints
The decomposition is applied on the pre-trained one stream VinVL [model](https://github.com/microsoft/Oscar/blob/master/VinVL_DOWNLOAD.md#pre-trained-models), so you need to download it first.
```
path/to/azcopy copy 'https://biglmdiag.blob.core.windows.net/vinvl/model_ckpts/TASK_NAME' coco_ir --recursive
```

Afterwards, you can run our code to perform decomposition.

You can also directly use our pre-trained checkpoints for [Flickr30k](https://drive.google.com/file/d/1nL1GUj62TssgRO34SoHwKKXVFrXRktrw/view?usp=sharing) and [COCO](https://drive.google.com/file/d/1nL1GUj62TssgRO34SoHwKKXVFrXRktrw/view?usp=sharing).

# Running
Run contrastive_learn.py using following args:
```
"program": "${workspaceFolder}/base.py",
"args": [
    "--tagslabel",
    "/data/wuhongfa/hashing/data/coco/img_tagslabel.json",
    "--img_feat_file",
    "/data/wuhongfa/hashing/data/coco/model_0060000/features.tsv",
    "--do_lower_case",
    "--output_dir",
    "output/coco_base",
    "--eval_model_dir",
    "/data/wuhongfa/hashing/data/checkpoint/checkpoint-1340000/",
    "--training_size","10000",
    "--query_size","5000",
    "--database_size","20000",
    "--class_number","80",
    "--class_name","/data/wuhongfa/hashing/data/coco/class_name.json",
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
    "--tagslabel",
    "/data/wuhongfa/hashing/data/coco/img_tagslabel.json",
    "--img_feat_file",
    "/data/wuhongfa/hashing/data/coco/model_0060000/features.tsv",
    "--do_lower_case",
    "--output_dir",
    "output/coco_base",
    "--eval_model_dir",
    "/data/wuhongfa/hashing/data/checkpoint/checkpoint-1340000/",
    "--training_size","10000",
    "--query_size","5000",
    "--database_size","20000",
    "--class_number","80",
    "--class_name","/data/wuhongfa/hashing/data/coco/class_name.json",
    "--bit_num","64",
    "--batch_size","128",
    "--test",
]
```


# Acknowledge
This repo is modified based on the [VinVL](https://github.com/microsoft/Oscar), we thank the authors for sharing their project.
