from torch.utils.data import Dataset
from oscar.utils.tsv_file import TSVFile
import base64
import os.path as op
import torch
import json
import os
from tqdm import tqdm
import numpy as np
import random
class  RetrievalDataset(Dataset):
    """ Image/Text Retrieval Dataset"""
    def __init__(self, tokenizer, args, split='train', is_train=True):
        super(RetrievalDataset, self).__init__()
        self.data_dir = args.data_dir
        self.img_file = args.img_feat_file #the vinvl feature from image
        caption_file = op.join(self.data_dir, '{}_captions.pt'.format(split))
        self.img_tsv = TSVFile(self.img_file)
        self.captions = torch.load(caption_file)
        self.img_keys = list(self.captions.keys())  # img_id as int
        if not type(self.captions[self.img_keys[0]]) == list:
            self.captions = {k: json.loads(self.captions[k]) for k in self.img_keys}
        # get the image image_id to index map
        imgid2idx_file = op.join(op.dirname(self.img_file), 'imageid2idx.json')
        self.image_id2idx = json.load(open(imgid2idx_file))  # img_id as string 
        if args.add_od_labels:
            label_data_dir = op.dirname(self.img_file)
            label_file = os.path.join(label_data_dir, "predictions.tsv")
            self.label_tsv = TSVFile(label_file)
            self.labels = {}
            for line_no in tqdm(range(self.label_tsv.num_rows())):
                row = self.label_tsv.seek(line_no)
                image_id = row[0]
                if int(image_id) in self.img_keys:
                    results = json.loads(row[1])
                    objects = results['objects'] if type(
                        results) == dict else results
                    self.labels[int(image_id)] = {
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
        if(is_train):
            self.num_captions_per_img = args.num_captions_per_img_train
        else:
            self.num_captions_per_img = args.num_captions_per_img_val
            if args.eval_img_keys_file:
                # select a subset of image keys for evaluation. eg. COCO 1k and 5k
                # eval_img_keys_file is a list of image keys saved in tsv file
                with open(op.join(args.data_dir, args.eval_img_keys_file), 'r') as f:
                    img_keys = f.readlines()
                self.img_keys = [int(k.strip()) for k in img_keys]
                self.captions = {k: self.captions[k] for k in self.img_keys}
                if args.add_od_labels:
                    self.labels = {k: self.labels[k] for k in self.img_keys}
        self.is_train = is_train
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.max_img_seq_length = args.max_img_seq_length
        self.add_od_labels = args.add_od_labels#att_mask_type
        self.att_mask_type = args.att_mask_type
    def get_image_caption_index(self, index):
        if(self.is_train):
            img_idx = index 
            cap_idx = np.random.choice(self.num_captions_per_img)
            return img_idx, [self.img_keys[img_idx], cap_idx]            
        else:
            img_idx = index // self.num_captions_per_img
            cap_idx = index % self.num_captions_per_img
            return img_idx, [self.img_keys[img_idx], cap_idx]
    def get_od_labels(self, img_key):
        if self.add_od_labels:
            if type(self.labels[img_key]) == str:
                od_labels = self.labels[img_key]
            else:
                od_labels = ' '.join(self.labels[img_key]['class'])
            return od_labels
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
        #a padding
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
        #merge
        input_ids = input_ids+input_ids_b
        segment_ids = segment_ids+segment_ids_b
        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_length:
            img_feat = img_feat[0 : self.max_img_seq_length, :]
            img_len = img_feat.shape[0]
            img_padding_len = 0
        else:
            img_padding_len = self.max_img_seq_length - img_len
            padding_matrix = torch.zeros((img_padding_len, img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # generate attention_mask
        att_mask_type = self.att_mask_type
        test_ = self.tokenizer.decode(input_ids)
        if att_mask_type == "CLR":
            attention_mask = [1] * seq_len_a + [0] * seq_padding_len_a +[1] * seq_len_b + [0] * seq_padding_len_b +  [1] * img_len + [0] * img_padding_len 
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        return (input_ids, attention_mask, segment_ids, img_feat)


    def __getitem__(self, index):

        img_idx, cap_idxs = self.get_image_caption_index(index)
        img_key = self.img_keys[img_idx]
        feature = self.get_image(img_key)
        caption = self.captions[cap_idxs[0]][cap_idxs[1]]
        od_labels= self.get_od_labels(img_key)
        example = self.tensorize_example(caption, feature, text_b=od_labels)
        return index, example

    def get_test_labels(self):
        all_image = len(self.img_keys)
        all_sample = len(self.img_keys)*self.num_captions_per_img
        #all_label = self.num_captions_per_img
        label = []
        #where index is 1
        for i in tqdm(range(all_image)):
            start = i*self.num_captions_per_img
            label_i = [0 for i in range(all_sample)]
            for j in range(start,start+self.num_captions_per_img):
                label_i[j]=1
            label.extend(label_i)
        return label
    
    def get_image(self, image_id):
        image_idx = self.image_id2idx[str(image_id)]
        row = self.img_tsv.seek(image_idx)
        num_boxes = int(row[1])
        features = np.frombuffer(base64.b64decode(row[-1]),
                                 dtype=np.float32).reshape((num_boxes, -1)).copy()
        t_features = torch.from_numpy(features)
        return t_features

    def __len__(self):
        if(self.is_train):
            return len(self.img_keys) 
        else:
            return len(self.img_keys) * self.num_captions_per_img
class  RetrievalDatasetSEP(Dataset):
    """ Image/Text Retrieval Dataset"""
    def __init__(self, tokenizer, args, split='train', is_train=True):
        super(RetrievalDatasetSEP, self).__init__()
        self.data_dir = args.data_dir
        self.img_file = args.img_feat_file #the vinvl feature from image
        caption_file = op.join(self.data_dir, '{}_captions.pt'.format(split))
        self.img_tsv = TSVFile(self.img_file)
        self.captions = torch.load(caption_file)
        self.img_keys = list(self.captions.keys())  # img_id as int
        if not type(self.captions[self.img_keys[0]]) == list:
            self.captions = {k: json.loads(self.captions[k]) for k in self.img_keys}
        # get the image image_id to index map
        imgid2idx_file = op.join(op.dirname(self.img_file), 'imageid2idx.json')
        self.image_id2idx = json.load(open(imgid2idx_file))  # img_id as string 
        if args.add_od_labels:
            label_data_dir = op.dirname(self.img_file)
            label_file = os.path.join(label_data_dir, "predictions.tsv")
            self.label_tsv = TSVFile(label_file)
            self.labels = {}
            for line_no in tqdm(range(self.label_tsv.num_rows())):
                row = self.label_tsv.seek(line_no)
                image_id = row[0]
                if int(image_id) in self.img_keys:
                    results = json.loads(row[1])
                    objects = results['objects'] if type(
                        results) == dict else results
                    self.labels[int(image_id)] = {
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
        if(is_train):
            self.num_captions_per_img = args.num_captions_per_img_train
        else:
            self.num_captions_per_img = args.num_captions_per_img_val
            if args.eval_img_keys_file:
                # select a subset of image keys for evaluation. eg. COCO 1k and 5k
                # eval_img_keys_file is a list of image keys saved in tsv file
                with open(op.join(args.data_dir, args.eval_img_keys_file), 'r') as f:
                    img_keys = f.readlines()
                self.img_keys = [int(k.strip()) for k in img_keys]
                self.captions = {k: self.captions[k] for k in self.img_keys}
                if args.add_od_labels:
                    self.labels = {k: self.labels[k] for k in self.img_keys}
        self.is_train = is_train
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.max_img_seq_length = args.max_img_seq_length
        self.add_od_labels = args.add_od_labels#att_mask_type
        self.att_mask_type = args.att_mask_type
    def get_image_caption_index(self, index):
        if(self.is_train):
            img_idx = index 
            cap_idx = np.random.choice(self.num_captions_per_img)
            return img_idx, [self.img_keys[img_idx], cap_idx]            
        else:
            img_idx = index // self.num_captions_per_img
            cap_idx = index % self.num_captions_per_img
            return img_idx, [self.img_keys[img_idx], cap_idx]
    def get_od_labels(self, img_key):
        if self.add_od_labels:
            if type(self.labels[img_key]) == str:
                od_labels = self.labels[img_key]
            else:
                od_labels = ' '.join(self.labels[img_key]['class'])
            return od_labels
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
        #a padding
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
        #merge
        input_ids = input_ids+input_ids_b
        segment_ids = segment_ids+segment_ids_b
        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_length:
            img_feat = img_feat[0 : self.max_img_seq_length, :]
            img_len = img_feat.shape[0]
            img_padding_len = 0
        else:
            img_padding_len = self.max_img_seq_length - img_len
            padding_matrix = torch.zeros((img_padding_len, img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # generate attention_mask
        att_mask_type = self.att_mask_type
        test_ = self.tokenizer.decode(input_ids)
        if att_mask_type == "CLR":
            attention_mask = [1] * seq_len_a + [0] * seq_padding_len_a +[1] * seq_len_b + [0] * seq_padding_len_b +  [1] * img_len + [0] * img_padding_len 
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        a_token_len = torch.tensor([len(tokens_a) + 1], dtype=torch.long)
        b_token_len = torch.tensor([len(tokens_b) ], dtype=torch.long)
        return (input_ids, attention_mask, segment_ids, img_feat,a_token_len,b_token_len)


    def __getitem__(self, index):

        img_idx, cap_idxs = self.get_image_caption_index(index)
        img_key = self.img_keys[img_idx]
        feature = self.get_image(img_key)
        caption = self.captions[cap_idxs[0]][cap_idxs[1]]
        od_labels= self.get_od_labels(img_key)
        example = self.tensorize_example(caption, feature, text_b=od_labels)
        return index, example

    def get_test_labels(self):
        all_image = len(self.img_keys)
        all_sample = len(self.img_keys)*self.num_captions_per_img
        #all_label = self.num_captions_per_img
        label = []
        #where index is 1
        for i in tqdm(range(all_image)):
            start = i*self.num_captions_per_img
            label_i = [0 for i in range(all_sample)]
            for j in range(start,start+self.num_captions_per_img):
                label_i[j]=1
            label.extend(label_i)
        return label
    
    def get_image(self, image_id):
        image_idx = self.image_id2idx[str(image_id)]
        row = self.img_tsv.seek(image_idx)
        num_boxes = int(row[1])
        features = np.frombuffer(base64.b64decode(row[-1]),
                                 dtype=np.float32).reshape((num_boxes, -1)).copy()
        t_features = torch.from_numpy(features)
        return t_features

    def __len__(self):
        if(self.is_train):
            return len(self.img_keys) 
        else:
            return len(self.img_keys) * self.num_captions_per_img

class IRRetrievalDataset(Dataset):
    """ Image/Text Retrieval Dataset"""
    def __init__(self,image_keys, args,tokenizer,train=False):
        """
        tokenizer: tokenizer to process caption text.
        args: configureation parameters including max_seq_length, etc.
        split: used to infer the data used for training or testing. 
             All files are in .pt format of a dictionary with image keys and 
             image features (pytorch tensors), captions (list of str, support multiple
             captions per image), labels (list of dictionary or str of all labels),

        """
        super(IRRetrievalDataset, self).__init__()
        with open(args.tagslabel ,"r") as f:
            self.tagslabel = json.load(f)
        label_tsv =TSVFile(args.img_feat_file)
        self.ob_labels = {}
        self.img_keys = image_keys
        if(train):
            self.img_keys = self.img_keys[args.divide:]

        for line_no in tqdm(range(label_tsv.num_rows())):
            row = label_tsv.seek(line_no)
            image_id = row[0]
            results = json.loads(row[1])
            objects = results['objects'] if type(results) == dict else results
            self.ob_labels[image_id] = {
                "class": [cur_d['class'] for cur_d in objects],
                "boxes": np.array([cur_d['rect'] for cur_d in objects],dtype=np.float32),
                "feature":np.array([np.frombuffer(base64.b64decode(cur_d["feature"]),dtype=np.float32) for cur_d in objects],dtype=np.float32)
            }         
        self.output_mode = 'classification'
        self.tokenizer = tokenizer
        self.max_seq_length = 70
        self.max_img_seq_len = 70
        self.att_mask_type = "CLR"
    def get_od_labels(self, img_key):

        if type(self.ob_labels[img_key]) == str:
            od_labels = self.ob_labels[img_key]
        else:
            od_labels = ' '.join(self.ob_labels[img_key]['class'])
        return od_labels
        
    def tensorize_example(self, text_a, img_feat, text_b=None, 
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):
        if(not isinstance(text_a,list)):
            tokens_a = self.tokenizer.tokenize(text_a)
        else:
            tokens_a=text_a
        if len(tokens_a) > self.max_seq_length - 2:
            tokens_a = tokens_a[:(self.max_seq_length - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_a) + 1)
        seq_a_len = len(tokens)
        if text_b:
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_length - len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_length - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens)
        seq_padding_len = self.max_seq_length - seq_len
        tokens += [self.tokenizer.pad_token] * seq_padding_len
        segment_ids += [pad_token_segment_id] * seq_padding_len
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

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
        att_mask_type = self.att_mask_type
        if att_mask_type == "CLR":
            attention_mask = [1] * seq_len + [0] * seq_padding_len + \
                             [1] * img_len + [0] * img_padding_len
        else:
            # use 2D mask to represent the attention
            max_len = self.max_seq_length + self.max_img_seq_len
            attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
            # full attention of C-C, L-L, R-R
            c_start, c_end = 0, seq_a_len
            l_start, l_end = seq_a_len, seq_len
            r_start, r_end = self.max_seq_length, self.max_seq_length + img_len
            attention_mask[c_start : c_end, c_start : c_end] = 1
            attention_mask[l_start : l_end, l_start : l_end] = 1
            attention_mask[r_start : r_end, r_start : r_end] = 1
            if att_mask_type == 'CL':
                attention_mask[c_start : c_end, l_start : l_end] = 1
                attention_mask[l_start : l_end, c_start : c_end] = 1
            elif att_mask_type == 'CR':
                attention_mask[c_start : c_end, r_start : r_end] = 1
                attention_mask[r_start : r_end, c_start : c_end] = 1
            elif att_mask_type == 'LR':
                attention_mask[l_start : l_end, r_start : r_end] = 1
                attention_mask[r_start : r_end, l_start : l_end] = 1
            else:
                raise ValueError("Unsupported attention mask type {}".format(att_mask_type))
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        return (input_ids, attention_mask, segment_ids, img_feat)
    def getCaption(self,tag_list):
        if(isinstance(tag_list,list)):
            caption = ""
            for i in tag_list:
                caption+=i+" "
            caption=caption.strip()
        else:#is a string
            caption  = tag_list
        return caption
    def __getitem__(self, index):
        #img_idx, cap_idxs = self.get_image_caption_index(index)
        img_key = self.img_keys[index]
        feature = self.get_image(img_key)
        caption = self.tagslabel[img_key]["tags"]
        od_labels = self.get_od_labels(img_key)
        example = self.tensorize_example(caption, feature, text_b=od_labels)

        # select a negative pair
        neg_img_indexs = list(range(0, index)) + list(range(index + 1, len(self.img_keys)))
        img_idx_neg = random.choice(neg_img_indexs)
        

        if random.random() <= 0.5:
            # randomly select a negative caption from a different image.
            #cap_idx_neg = random.randint(0, self.num_captions_per_img - 1)
            img_key_neg = self.img_keys[index]
            if(img_key=="im19444"):
                print()
            caption_neg = self.tagslabel[img_key_neg]["tags"]
            example_neg = self.tensorize_example(caption_neg, feature, text_b=od_labels)

        else:
            # randomly select a negative image 
            feature_neg = self.get_image(self.img_keys[img_idx_neg])
            od_labels_neg = self.get_od_labels(self.img_keys[img_idx_neg])
            example_neg = self.tensorize_example(caption, feature_neg, text_b=od_labels_neg)

        example_pair = tuple(list(example) + [1] + list(example_neg) + [0])
        if(example_pair[0].shape[0]!=70):
            print()
        if(example_pair[5].shape[0]!=70):
            print()
        """"
        print(example_pair[0].shape)
        print(example_pair[1].shape)
        print(example_pair[2].shape)
        print(example_pair[3].shape)
        print(example_pair[4])
        print(example_pair[5].shape)
        print(example_pair[6].shape)
        print(example_pair[7].shape)
        print(example_pair[8].shape)
        print(example_pair[9])
        print(index)
        print("=============================================================")
        """
        
        return index, example_pair


    def get_image(self, image_id):
        image=  self.ob_labels[image_id]["feature"]

        boxes = self.ob_labels[image_id]["boxes"]

        height = self.ob_labels[image_id]["boxes"][:,0]-self.ob_labels[image_id]["boxes"][:,2]

        weight = self.ob_labels[image_id]["boxes"][:,1]-self.ob_labels[image_id]["boxes"][:,3]
        
        weight = weight[:,np.newaxis]

        height = height[:,np.newaxis]

        feature = np.concatenate((image,boxes,height,weight),axis=1)

        return torch.from_numpy(feature)

    def __len__(self):
        return len(self.img_keys)
