import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from logging import log
import re
import numpy as np
import os.path as op
from hashing_module.triplet_loss import *
from torch.autograd.grad_mode import F

from torch.nn.modules import loss
from torch.utils.data.sampler import Sampler

import argparse
from oscar.modeling.modeling_bert import HashingformerALL,normal_label
from pytorch_transformers import BertTokenizer, BertConfig
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from torch.autograd import Variable
from oscar.utils.tsv_file import TSVFile
from torch.nn import CrossEntropyLoss
import json
import base64
import random
from pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
from hashing_module.utils import calc_map_k
from oscar.utils.logger import setup_logger
from oscar.utils.misc import mkdir
from torch.nn import functional as F

def save_pretrained(model, save_directory,name="model.cpkt"):
    """ Save a model and its configuration file to a directory, so that it
        can be re-loaded using the `:func:`~pytorch_transformers.PreTrainedModel.from_pretrained`` class method.
    """
    assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

    # Only save the model it-self if we are using distributed training
    model_to_save = model.module if hasattr(model, 'module') else model

    # Save configuration file
    model_to_save.config.save_pretrained(save_directory)

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(save_directory, name)

    torch.save(model_to_save.state_dict(), output_model_file)
class Opt():
    def __init__(self) -> None:
        self.use_gpu = True
        self.training_size = 10000
        self.query_size = 2000
        self.bit = 64
        self.database_size = 18000 
        self.gamma = 1
        self.eta = 1
        self.valid = True
        self.batch_size = 64
        self.margin = 0.4
        self.gamma = 1
        self.beta = 1
        self.alpha = 1
opt = Opt()
class  ConstrastiveRetrievalDataset(Dataset):
    """ Image/Text Retrieval Dataset"""
    def __init__(self, train_L, batch_size):
        super(ConstrastiveRetrievalDataset, self).__init__()

        self.train_L=train_L.to(torch.float32).detach().cpu()
        self.data_len = train_L.shape[0]
        self.all_index = np.arange(self.data_len)
        np.random.shuffle(self.all_index)
        self.batch_size = batch_size

    def __getitem__(self, index):
        ind = self.all_index[index*self.batch_size: (index+1)*self.batch_size ]
        ind = torch.from_numpy(ind)
        return ind

    def __len__(self):

        return self.data_len//self.batch_size

def set_random_seed(seed):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_random_seed(1000)
class RetrievalDataset(Dataset):
    """ Image/Text Retrieval Dataset"""
    def __init__(self, args,tokenizer,split="train"):
        """
        tokenizer: tokenizer to process caption text.
        args: configureation parameters including max_seq_length, etc.
        split: used to infer the data used for training or testing. 
             All files are in .pt format of a dictionary with image keys and 
             image features (pytorch tensors), captions (list of str, support multiple
             captions per image), labels (list of dictionary or str of all labels),

        """
        super(RetrievalDataset, self).__init__()
        with open(args.tagslabel ,"r") as f:
            self.tagslabel = json.load(f)   
        self.args = args  
        self.split = split      
        self.img_file = args.img_feat_file
        self.img_tsv = TSVFile(self.img_file)
        self.img_keys = list(self.tagslabel.keys())  # img_id as int
        imgid2idx_file = op.join(op.dirname(self.img_file), 'imageid2idx.json')
        self.image_id2idx = json.load(open(imgid2idx_file))  # img_id as string
        with open(args.class_name,"r") as f:
            self.class_name = json.load(f)
            self.class_name = np.array(self.class_name)
        #if(args.split_keys):
        if(False):
            with open(args.split_keys,"r") as f:
                self.img_keys = json.load(f)
                self.img_keys = [str(i) for i in self.img_keys]
        else:
            random.seed(279834)
            random.shuffle(self.img_keys)
        if(split=="train"):
            self.img_keys = self.img_keys[args.query_size:args.training_size + args.query_size]
            total_label = []
            for i in self.img_keys:
                total_label.append(self.tagslabel[i]["label"])
            self.total_label = torch.Tensor(total_label)
        elif(split=="query"):
            self.img_keys = self.img_keys[:args.query_size]
        else:
            self.img_keys = self.img_keys[args.query_size:args.database_size+ args.query_size]
        label_data_dir = op.dirname(self.img_file)
        label_file = os.path.join(label_data_dir, "label.tsv")
        self.label_tsv = TSVFile(label_file)
        self.labels = {}

        
        for line_no in tqdm(range(self.label_tsv.num_rows())):
            row = self.label_tsv.seek(line_no)
            image_id = row[0]
            if image_id in self.img_keys:
                results = json.loads(row[1])
                objects = results['objects'] if type(
                    results) == dict else results
                self.labels[image_id] = {
                    "image_h": results["image_h"] if type(
                        results) == dict else 600,
                    "image_w": results["image_w"] if type(
                        results) == dict else 800,
                    "class": [cur_d['class'] for cur_d in objects],
                    "boxes": np.array([cur_d['rect'] for cur_d in objects],
                                        dtype=np.float32)
                }
        self.label_tsv._fp.close()
        self.label_tsv._fp = None   
        self.output_mode = 'classification'
        self.tokenizer = tokenizer
        self.max_seq_length = 35
        self.max_img_seq_len = 70
        self.args.max_label_length = args.max_label_length
    def get_od_labels(self, img_key):

        if type(self.labels[img_key]) == str:
            od_labels = self.labels[img_key]
        else:
            od_labels = ' '.join(self.labels[img_key]['class'])
        return od_labels
    def class_tokenize(self,labels,max_length=15):
        all_size = labels.shape[0]
        final_label = []
        for i in range(all_size):
            this_label = torch.zeros((max_length+2))
            class_name = self.class_name[labels[i]>0]
            
            tokens = self.tokenizer.tokenize("".join(class_name))
            tokens = [self.tokenizer.cls_token] + tokens[0:max_length] + [self.tokenizer.sep_token]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            this_label[0:len(input_ids)] = torch.Tensor(input_ids)
            final_label.append(this_label)
        final_label = torch.stack(final_label).long()  
        return final_label  

   
    def tensorize_example(self, text_a, img_feat, text_b=None, 
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):
        tokens_a = self.tokenizer.tokenize(text_a)
        if len(tokens_a) > self.max_seq_length - 2:
            tokens_a = tokens_a[:(self.max_seq_length - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_a) + 1)
    
        if text_b:
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_length   - 2:#a
                tokens_b = tokens_b[: (self.max_seq_length  - 2)]
            tokens_b = [self.tokenizer.cls_token] +tokens_b+ [self.tokenizer.sep_token]
            segment_ids_b = [sequence_b_segment_id] + [sequence_b_segment_id] * (len(tokens_b) -1)
        #这儿分a padding
        seq_len_a = len(tokens)
        seq_padding_len_a = self.max_seq_length - seq_len_a
        tokens += [self.tokenizer.pad_token] * seq_padding_len_a
        segment_ids += [pad_token_segment_id] * seq_padding_len_a
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        #b padding
        seq_len_b = len(tokens_b)
        seq_padding_len_b = self.max_seq_length - seq_len_b
        tokens_b += [self.tokenizer.pad_token] * seq_padding_len_b
        segment_ids_b += [pad_token_segment_id] * seq_padding_len_b
        input_ids_b = self.tokenizer.convert_tokens_to_ids(tokens_b)
        #合并
        input_ids = input_ids+input_ids_b
        segment_ids = segment_ids+segment_ids_b
        # image features
        img_len = img_feat.shape[0]

        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0 : self.max_img_seq_len, :]
            img_len = img_feat.shape[0]
            img_padding_len = 0
        else:
            img_padding_len = self.max_img_seq_len - img_len
            padding_matrix = torch.zeros((img_padding_len, img_feat.shape[1]))
            
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # generate attention_mask
        att_mask_type = "CLR"
        if att_mask_type == "CLR":
            attention_mask = [1] * seq_len_a + [0] * seq_padding_len_a +[1] * seq_len_b + [0] * seq_padding_len_b +  [1] * img_len + [0] * img_padding_len 


        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        return (input_ids, attention_mask, segment_ids, img_feat)

    def __getitem__(self, index):
        
        img_key = self.img_keys[index]
      
        feature = self.get_image(img_key)
        tag_list = self.tagslabel[img_key]["tags"]
        if(isinstance(tag_list,list)):
            caption = ""
            for i in tag_list:
                caption+=i+" "
            caption=caption.strip()
        else:#is a string
            caption  = tag_list
        od_labels = self.get_od_labels(img_key)
        example = self.tensorize_example(caption, feature, text_b=od_labels)
        label = self.tagslabel[img_key]["label"]
        label=torch.tensor(label, dtype=torch.long)
        if(self.split=="train"):
            raw_label = self.generate_samples(label,self.args.negative_number)
            #negative_label = normal_label(raw_label,max_length = self.args.max_label_length )
            negative_label = self.class_tokenize(raw_label,max_length = self.args.max_label_length )
            return tuple(list(example) + [label,negative_label,raw_label]),img_key
        else:
            return tuple(list(example) + [label]),img_key

    def generate_samples(self, label,negative_number = 99):
        mask = 1-label #
        negative_samples = []
        positive_sample = label 
        all_index =torch.arange(self.total_label.shape[0])
        while(len(negative_samples)<negative_number//2):
            smaples=  torch.from_numpy(np.random.choice(2, self.args.class_number,p=[1-5/self.args.class_number,5/self.args.class_number]))
            is_positive = (smaples*label).sum()>0
            if(not is_positive):
                negative_samples.append(smaples)
        
        while(len(negative_samples)<negative_number):
            sim = torch.matmul(label.unsqueeze(0).float(),self.total_label.float().t()).squeeze(0)>0
            negative = all_index[~sim]
            smaples_index=  np.random.choice(negative.numpy(), 1)[0]
            negative_samples.append(self.total_label[smaples_index])        
        final_sample = [positive_sample]+negative_samples
        final_sample = torch.stack(final_sample)
        return final_sample
    def get_image(self, image_id):
        image_idx = self.image_id2idx[str(image_id)]
        row = self.img_tsv.seek(image_idx)
        num_boxes = int(row[1])
        features = np.frombuffer(base64.b64decode(row[-1]),
                                 dtype=np.float32).reshape((num_boxes, -1)).copy()
        t_features = torch.from_numpy(features)
        return t_features

    def __len__(self):
        return len(self.img_keys) 

def calc_neighbor(label1, label2):
    # calculate the similar matrix
    label1=label1.to(torch.float32)
    label2=label2.to(torch.float32)
    Sim = (label1.matmul(label2.transpose(0, 1)) > 0)
    return Sim
def save_checkpoint(model, tokenizer, args, epoch):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}'.format(
        epoch))
    mkdir(checkpoint_dir)
    #model_to_save = model.module if hasattr(model, 'module') else model
    save_num = 0
    while (save_num < 10):
        try:
            
            save_pretrained(model=model,save_directory=checkpoint_dir)
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            save_num += 1
    if save_num == 10:
        logger.info("Failed to save checkpoint after 10 trails.")
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_epochs", default=1000, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--class_name", default='/raid/data_modal/MIR_Flickr_25k/class_name.json"', type=str, required=False,
                        help="The input data dir with all required files.")
    parser.add_argument("--split_keys", default='', type=str, required=False,
                        help="split_keys")
    parser.add_argument("--tagslabel", default='MIR_Flickr_25k/img_tagslabel.json', type=str, required=False,
                        help="The input data dir with all required files.")
    parser.add_argument("--img_feat_file", default='/MIR_Flickr_25k/vinvl_data/vinvl_vg_x152c4/predictions.tsv', type=str, required=False,
                        help="The absolute address of the image feature file.")
    parser.add_argument("--output_dir", default='output/log_aipr', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--num_workers", default=4, type=int, help="Workers in dataloader.")
    parser.add_argument("--eval_model_dir", type=str, default='', 
                        help="Model directory for evaluation.")   
    parser.add_argument("--do_lower_case", action='store_true', 
                        help="Set this flag if you are using an uncased model.")   
    parser.add_argument("--output_file", type=str, default='', 
                        help="Model directory for evaluation.")  
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear.")
    parser.add_argument("--bit", default=64, type=int, help="constant or linear.")
    parser.add_argument("--class_number", default=255, type=int, help="constant or linear.")  
    parser.add_argument("--training_size", default=10000, type=int, help="constant or linear.") 
    parser.add_argument("--query_size", default=2000, type=int, help="constant or linear.") 
    parser.add_argument("--database_size", default=18000, type=int, help="constant or linear.")
    parser.add_argument("--no_pretrain", action='store_true', help="constant or linear.")
    parser.add_argument("--negative_number", default=9, type=int, help="constant or linear.")
    parser.add_argument("--max_label_length", default=10, type=int, help="constant or linear.")
    parser.add_argument("--temperature", default=0.05, type=float, help="constant or linear.")
    args = parser.parse_args()
    if(args.training_size != -1):
        opt.training_size = args.training_size
        opt.query_size = args.query_size
        opt.database_size = args.database_size
    global logger
    mkdir(args.output_dir)
    logger = setup_logger("vlpretrain", args.output_dir, 0)
    opt.bit = args.bit
    opt.class_number = args.class_number
    device = torch.device("cuda")
    config_class, tokenizer_class = BertConfig, BertTokenizer
    checkpoint = args.eval_model_dir
    tokenizer = tokenizer_class.from_pretrained(checkpoint, do_lower_case=args.do_lower_case)
    config = config_class.from_pretrained(checkpoint)
    config.class_number = opt.class_number
    config.bit = args.bit
    model = HashingformerALL(None,config)
    #sd = torch.load(checkpoint+"/model.cpkt", map_location="cpu")
    if(not args.no_pretrain):

        if(not os.path.exists(checkpoint+"/pytorch_model.bin")):
            sd = torch.load(checkpoint+"/model.cpkt", map_location="cpu")
        else:
            sd = torch.load(checkpoint+"/pytorch_model.bin", map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)  
    model.to(device)

    train_dataset =  RetrievalDataset(args,tokenizer,"train")
    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
            batch_size=opt.batch_size, num_workers=4)

    query_dataset = RetrievalDataset(args,tokenizer,"query")
    retrieval_dataset = RetrievalDataset(args,tokenizer,"retrieval")



    #optimizer
    t_total = opt.training_size* args.num_train_epochs
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not \
            any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if \
            any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.scheduler == "constant":
        scheduler = WarmupConstantSchedule(
                optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = WarmupLinearSchedule(
                optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))
    
    
    max_mapi2t = 0
    max_mapt2i = 0
    #新建数据库
    batch_size = opt.batch_size
    crossEntropyLoss = torch.nn.CrossEntropyLoss(reduction="sum")
    tri_loss = TripletLoss(opt)
    for epoch in range(args.num_train_epochs):   
        model.train()

        global_acc_sim = 0
        global_contrastive_t = 0
        global_contrastive_i = 0 
        contrastive_loss = 0
        count  = 0 
        for batch,img_key in tqdm(train_dataloader):
            #this_train_L = Variable(train_L[query_index])   
            count+=1
            batch_size = batch[0].shape[0]
            train_input_ids_this = batch[0]
            train_attention_mask_this = batch[1]
            train_token_type_ids_this = batch[2]
            train_img_feats_this = batch[3]
            sample_L = batch[4]
            negative_sample = batch[5]
            raw_label = batch[6]
            if opt.use_gpu:
                sample_L = sample_L.float().cuda()
                negative_sample = negative_sample.long().cuda()
                negative_sample = negative_sample.reshape(-1,args.max_label_length+2)
                train_input_ids_this = train_input_ids_this.cuda()
                train_attention_mask_this =train_attention_mask_this.cuda()
                train_token_type_ids_this =train_token_type_ids_this.cuda()
                train_img_feats_this =train_img_feats_this.cuda()
                raw_label = raw_label.float().cuda()
                contrastive_label = torch.zeros(batch_size).long().cuda()
                one_hot = torch.zeros(batch_size, args.negative_number+1).cuda().scatter_(1, contrastive_label.unsqueeze(1), 1)
            #tets = torch.gather(S,1,retrieval_index)
            hashing_i = model(input_ids=train_input_ids_this,token_type_ids=train_token_type_ids_this,
                                    attention_mask=train_attention_mask_this,img_feats=train_img_feats_this,modal="i")
            hashing_t  = model(input_ids=train_input_ids_this,token_type_ids=train_token_type_ids_this,
                                    attention_mask=train_attention_mask_this,img_feats=train_img_feats_this,modal="t")
            hashing_label = model(input_ids=negative_sample,modal="label")
            hashing_label = hashing_label.reshape(batch_size,-1,args.bit)
            #分类损失
            logit = torch.matmul(hashing_i,hashing_t.t())
            sim = torch.matmul(sample_L,sample_L.t())>0
            theta_it = 1/2*logit
            log_loss_it = -torch.sum(sim * theta_it - torch.log(1.0 + torch.exp(theta_it)))

            global_acc_sim+=f1_calc(logit,sim,0)   

             
            #对比学习损失
            i_contrastive_logit =torch.matmul(hashing_i.unsqueeze(1),hashing_label.transpose(1,2)).squeeze(1)
            t_contrastive_logit =torch.matmul(hashing_t.unsqueeze(1),hashing_label.transpose(1,2)).squeeze(1)
            #标签学习
            
            hashing_label_positive = hashing_label[:,0,:]
            new_hashing_label = hashing_label.clone()
            new_hashing_label[:,0,:] = hashing_i
            i_label_contrastive =torch.matmul(hashing_label_positive.unsqueeze(1),new_hashing_label.transpose(1,2)).squeeze(1)
            new_hashing_label = hashing_label.clone()
            new_hashing_label[:,0,:] = hashing_t
            t_label_contrastive =torch.matmul(hashing_label_positive.unsqueeze(1),new_hashing_label.transpose(1,2)).squeeze(1)

            
            loss_logit_i = torch.sigmoid(i_contrastive_logit/2)/args.temperature
            loss_logit_t = torch.sigmoid(t_contrastive_logit/2)/args.temperature
            contrastive_loss = crossEntropyLoss(loss_logit_i,contrastive_label) +crossEntropyLoss(loss_logit_t,contrastive_label)\
                                +crossEntropyLoss(i_label_contrastive,contrastive_label)+crossEntropyLoss(t_label_contrastive,contrastive_label)
            contrastive_loss = contrastive_loss
                    
            
            #准确率
            global_contrastive_i+=f1_calc(i_contrastive_logit,one_hot,0)
            global_contrastive_t+=f1_calc(t_contrastive_logit,one_hot,0)
            
            hashing_loss =contrastive_loss +log_loss_it 
            
            loss_x =  hashing_loss# +
            loss_x.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        global_acc_sim = global_acc_sim/count
        global_contrastive_i = global_contrastive_i/count
        global_contrastive_t = global_contrastive_t/count
        logger.info('...epoch: %3d, log_loss_x: %3.3f,%3.3f acc: %3.3f,%3.3f,%3.3f  lr: %f' 
        % (epoch + 1, hashing_loss,contrastive_loss,global_acc_sim,global_contrastive_i,global_contrastive_t,optimizer.param_groups[0]["lr"]))
        if opt.valid and epoch%2==0:
            mapi2t, mapt2i,mapi2t_real ,mapt2i_real,query_acc,log_loss= valid(model,query_dataset,retrieval_dataset)

            logger.info('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f, MAP_R(i->t): %3.4f, MAP_R(t->i): %3.4f  predict:%3.3f,%3.3f' % (epoch + 1, mapi2t, 
            mapt2i,mapi2t_real ,mapt2i_real,query_acc,log_loss))
            if mapt2i >= max_mapt2i and mapi2t >= max_mapi2t:
                max_mapi2t = mapi2t
                max_mapt2i = mapt2i
                save_checkpoint(model, tokenizer, args, epoch) 
            logger.info('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))              
def f1_calc(logit,sim,threshold):
    logit = logit.detach().cpu()
    sim = sim.detach().cpu()

    t = (sim>threshold).sum()
    if(t<1):
        return 0
    tp = (logit[sim>threshold]>threshold).sum()
    p = (logit>threshold).sum()
    
    if(tp>0 and p>0 and t>0):
        recall = tp/t
        precision = tp/p
        f1 = 2* recall*precision/(recall+precision)
        return f1
    return 0

def generate_code(model, query_dataloader,TorI="I"):

    class_logits = []
    labels = []
    hashing_bit = []
    for batch,keys in tqdm(query_dataloader):
        #image = X[ind]#.unsqueeze(1).unsqueeze(-1).type(torch.float)
        train_input_ids_this = batch[0].long().cuda()
        train_attention_mask_this = batch[1].long().cuda()
        train_token_type_ids_this = batch[2].long().cuda()
        train_img_feats_this = batch[3].cuda()
        label = batch[4].cuda()
        with torch.no_grad():
            if(TorI=="T"):
                cur_f= model(input_ids=train_input_ids_this,token_type_ids=train_token_type_ids_this,
                                        attention_mask=train_attention_mask_this,img_feats=train_img_feats_this,modal="t")
            else:
                cur_f= model(input_ids=train_input_ids_this,token_type_ids=train_token_type_ids_this,
                                        attention_mask=train_attention_mask_this,img_feats=train_img_feats_this,modal="i")
            hashing_bit.append(cur_f)
            labels.append(label)
    hashing_bit = torch.cat(hashing_bit,0)
    labels = torch.cat(labels,0)

    #B = torch.sign(B)
    return hashing_bit,labels

def valid(model, query_dataset,retrieval_dataset) :
    model.eval()
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler,
            batch_size=64, num_workers=4)
    retrieval_sampler = SequentialSampler(retrieval_dataset)
    retrieval_dataloader = DataLoader(retrieval_dataset, sampler=retrieval_sampler,
            batch_size=64, num_workers=4)
    qBX,query_L_i = generate_code(model,query_dataloader ,TorI="I")
    qBY,query_L_t = generate_code(model, query_dataloader, TorI="T")
    rBX,retrieval_L_i = generate_code(model, retrieval_dataloader,TorI="I")
    rBY,retrieval_L_t = generate_code(model, retrieval_dataloader, TorI="T")
    
    
    mapi2t_real = calc_map_k(qBX, rBY, query_L_i, retrieval_L_t)
    mapt2i_real = calc_map_k(qBY, rBX, query_L_t, retrieval_L_i)

    mapi2t = calc_map_k(torch.sign(qBX), torch.sign(rBY), query_L_i, retrieval_L_t)
    mapt2i = calc_map_k(torch.sign(qBY), torch.sign(rBX), query_L_t, retrieval_L_i)


    logit = torch.matmul(qBX,rBY.t())
    sim = torch.matmul(query_L_i.float(),retrieval_L_t.float().t())>0
    theta = 1/2*logit
    log_loss = -torch.mean(sim * theta - torch.log(1.0 + torch.exp(theta)))
    query_acc = f1_calc(logit,sim,0)


    return mapi2t, mapt2i,mapi2t_real ,mapt2i_real,query_acc,log_loss
if __name__ == '__main__':
    main()