import torch
from tqdm import tqdm
import numpy as np
def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH
def cal_confusion_gpu(hamm,gnd,pr_curve):
    #
    for threshold in range(1,len(pr_curve)+1):
        prediction =(hamm<threshold)
        confusion_vector = prediction / gnd 
        true_positives = torch.sum(confusion_vector == 1).item()
        false_positives = torch.sum(confusion_vector == float('inf')).item()
        true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
        false_negatives = torch.sum(confusion_vector == 0).item()

        pr_curve[threshold-1]["TP"]+=true_positives
        pr_curve[threshold-1]["FP"]+=false_positives
        pr_curve[threshold-1]["TN"]+=true_negatives
        pr_curve[threshold-1]["FN"]+=false_negatives 

def cal_confusion(hamm,gnd,pr_curve):
    #

    for threshold in range(1,len(pr_curve)+1):
        for ind,positive in enumerate(gnd):
            if(positive==1 and hamm[ind]>threshold):  #样本为正，且预测正确
                pr_curve[threshold-1]["TP"]+=1
            elif(positive==1 and hamm[ind]<threshold): #样本为正，且预测错误
                pr_curve[threshold-1]["FP"]+=1
            elif(positive==0 and hamm[ind]<threshold):#样本为负，且预测正确
                pr_curve[threshold-1]["TN"]+=1
            elif(positive==0 and hamm[ind]>threshold):#样本为负，且预测错误
                pr_curve[threshold-1]["FN"]+=1   
def cal_pr(pr_curve):
    for threshold in range(0,16):
        TP_FP = pr_curve[threshold]["TP"]+pr_curve[threshold]["FP"]
        TP_FN = pr_curve[threshold]["TP"]+pr_curve[threshold]["FN"]
        if(TP_FP!=0):
            pr_curve[threshold]["P"] = pr_curve[threshold]["TP"]/TP_FP
        else:
            pr_curve[threshold]["P"] = 0

        if(TP_FN!=0):
            pr_curve[threshold]["R"] = pr_curve[threshold]["TP"]/TP_FN
        else:
            pr_curve[threshold]["R"] = 0
def calc_map_k_final(qB, rB, query_L, retrieval_L, k=None):

    query_L = query_L.type(torch.float32)
    retrieval_L = retrieval_L.type(torch.float32)
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    map = 0
    bit = qB.shape[1]
    if k is None:
        k = retrieval_L.shape[0]
    print("calc map k")
    top10 = []
    pr_curve = []#16维
    for i in range(bit):
        pr_curve.append({"TP":0,"FP":0,"TN":0,"FN":0})

    for iter in tqdm(range(num_query)):

        q_L = query_L[iter]#取出相应q的label
        
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)#此query相似的label
        tsum = torch.sum(gnd)
        if tsum == 0:#有没有相似的
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        hamm_forpr = hamm.squeeze()
        gnd_forpr = gnd
        cal_confusion_gpu(hamm_forpr,gnd_forpr,pr_curve)
        _, ind = torch.sort(hamm)#按照概率来排大小
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
        map = map + torch.mean(count / tindex)
        top10_index = ind.detach().cpu().numpy()[0:10].tolist()
        top10_label = gnd.detach().cpu().numpy()[0:10].tolist()
        precison = sum(top10_label)/10
        top10.append((top10_index,top10_label,precison))
            
    cal_pr(pr_curve)
    map = map / num_query
    return map,top10,pr_curve
def calc_map_k(qB, rB, query_L, retrieval_L, k=None):
    query_L = query_L.type(torch.float32)
    retrieval_L = retrieval_L.type(torch.float32)
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    print("calc map k")



    for iter in tqdm(range(num_query)):

        q_L = query_L[iter]#取出相应q的label
        
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)#此query相似的label
        tsum = torch.sum(gnd)
        if tsum == 0:#有没有相似的
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        #hamm_forpr = hamm.squeeze()
        #gnd_forpr = gnd
        #cal_confusion_gpu(hamm_forpr,gnd_forpr,pr_curve)
        a, ind = torch.sort(hamm)#按照概率来排大小
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
        map = map + torch.mean(count / tindex)
        #top10_index = ind.detach().cpu().numpy()[0:10].tolist()
        #top10_label = gnd.detach().cpu().numpy()[0:10].tolist()
        #precison = sum(top10_label)/10
        #top10.append((top10_index,top10_label,precison))
            
    #cal_pr(pr_curve)
    map = map / num_query
    return map#,top10#,pr_curve
def calc_map_k_analysis(qB, rB, query_L, retrieval_L, k=None):
    query_L = query_L.type(torch.float32)
    retrieval_L = retrieval_L.type(torch.float32)
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    print("calc map k")



    for iter in tqdm(range(num_query)):

        q_L = query_L[iter]#取出相应q的label
        
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)#此query相似的label
        tsum = torch.sum(gnd)
        if tsum == 0:#有没有相似的
            continue
        hamm = torch.matmul(qB[iter, :].unsqueeze(0) , rB.t())
        #hamm_forpr = hamm.squeeze()
        #gnd_forpr = gnd
        #cal_confusion_gpu(hamm_forpr,gnd_forpr,pr_curve)
        _, ind = torch.sort(hamm,descending=True)#按照概率来排大小
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
        map = map + torch.mean(count / tindex)
        #top10_index = ind.detach().cpu().numpy()[0:10].tolist()
        #top10_label = gnd.detach().cpu().numpy()[0:10].tolist()
        #precison = sum(top10_label)/10
        #top10.append((top10_index,top10_label,precison))
            
    #cal_pr(pr_curve)
    map = map / num_query
    return map#,top10#,pr_curve

if __name__ == '__main__':
    qB = torch.Tensor([[1, -1, 1, 1],
                       [-1, -1, -1, 1],
                       [1, 1, -1, 1],
                       [1, 1, 1, -1]])
    rB = torch.Tensor([[1, -1, 1, -1],
                       [-1, -1, 1, -1],
                       [-1, -1, 1, -1],
                       [1, 1, -1, -1],
                       [-1, 1, -1, -1],
                       [1, 1, -1, 1]])
    query_L = torch.Tensor([[0, 1, 0, 0],
                            [1, 1, 0, 0],
                            [1, 0, 0, 1],
                            [0, 1, 0, 1]])
    retrieval_L = torch.Tensor([[1, 0, 0, 1],
                                [1, 1, 0, 0],
                                [0, 1, 1, 0],
                                [0, 0, 1, 0],
                                [1, 0, 0, 0],
                                [0, 0, 1, 0]])

    map = calc_map_k(qB, rB, query_L, retrieval_L)
    print(map)
