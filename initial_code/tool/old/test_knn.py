import time
import numpy as np
from sklearn.neighbors import KDTree

import torch

from knn_cuda import KNN
from lib.pointops.functions import pointops


def t2n(t):
    return t.detach().cpu().numpy()


def run_kdtree(ref, query, k):
    bs = ref.shape[0]
    D, I = [], []
    for j in range(bs):
        tree = KDTree(ref[j], leaf_size=100)
        d, i = tree.query(query[j], k=k)
        D.append(d)
        I.append(i)
    D = np.stack(D)
    I = np.stack(I)
    return D, I


def run_knnCuda(ref, query, k):
    ref = torch.from_numpy(ref).float().cuda()
    query = torch.from_numpy(query).float().cuda()
    knn = KNN(k, transpose_mode=True)
    d, i = knn(ref, query)
    return t2n(d), t2n(i)


def knnquery_mm(xyz1, xyz2, nsample):
    """
    KNN Indexing
    input: nsample: int32, Number of neighbor
           xyz1: (b, n, 3) coordinates of the features
           xyz2: (b, m, 3) centriods
        output: idx: (b, m, nsample)
    """
    r_xyz1 = torch.sum(xyz1 * xyz1, dim=2, keepdim=True)  # (B,N,1)
    r_xyz2 = torch.sum(xyz2 * xyz2, dim=2, keepdim=True)  # (B,M,1)
    mul = torch.matmul(xyz2, xyz1.permute(0,2,1))         # (B,M,N)
    dist = r_xyz2 - 2 * mul + r_xyz1.permute(0, 2, 1)       # (B,M,N)
    knn = dist.topk(k=nsample, dim=-1, largest=False)
    idx = knn.indices.int()
    return idx


if __name__ == "__main__":
    B, C, N, K = 4, 3, 4096*2, 32
    refer = torch.randn(B, N, C).cuda()
    #query = torch.randn(B, N, C).cuda()
    query = refer

    refer_c = refer.cpu().numpy()
    query_c = query.cpu().numpy()

    knn = KNN(K, transpose_mode=True)
    

    for i in range(20):
        torch.cuda.synchronize()
        t1 = time.time()
        #idx1 = knnquery_mm(refer, query, K)
        #_, idx2 = knn(refer, query)
        #_, idx3 = run_kdtree(refer_c, query_c, K)
        #idx4 = pointops.knnquery(K, refer, query)
        idx5 = pointops.knnquery_heap(K, refer, query)

        torch.cuda.synchronize()
        t2 = time.time()
        '''
        _, idx2 = knn(refer, query)

        torch.cuda.synchronize()
        t3 = time.time()
        _, idx3 = run_kdtree(refer_c, query_c, K)

        torch.cuda.synchronize()
        t4 = time.time()
        print(idx1[0,0:5],'\n'), print(idx2[0,0:5],'\n'), print(idx3[0,0:5],'\n')
        #assert np.sum(np.abs((idx1.cpu().numpy()-idx2.cpu().numpy()))) == 0
        #assert np.sum(np.abs((idx1.cpu().numpy()-idx3))) == 0
        print("knnquery_fast-{:.4f}, KNN_new-{:.4f}, run_kdtree-{:.4f}".format(t2-t1, t3-t2, t4-t3))
        '''
        print("knnquery-{:.4f}".format(t2-t1))
        #print(idx3[0,0:1],'\n'), print(idx4[0,0:1].cpu().numpy(),'\n'), print(idx5[0,0:1].cpu().numpy(),'\n')
        #a = np.sum(np.abs((idx4.cpu().numpy()-idx3)))
        #b = np.sum(np.abs((idx5.cpu().numpy()-idx3)))
        #c = np.sum(np.abs((idx5.cpu().numpy()-idx4.cpu().numpy())))
        #print("a={}, b={}, c={}".format(a, b, c))
        #assert a == 0
        #assert b == 0
