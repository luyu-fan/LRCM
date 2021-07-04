import torch as torch,random
import torch.nn as nn

class FDALoss(nn.Module):

    def __init__(self,sample_ratio = 0.3,prob_margin = 0.5):

        super(FDALoss,self).__init__()
        self.innner_cri = nn.MSELoss(reduction = "none")
        self.sample_ratio = sample_ratio
        self.prob_margin = prob_margin

    def forward(self,mixed_scores):

        """
        mixed_features: (batch * 2) * patches_num * embedding_dim
        """
        b,num,dim = mixed_scores.size()
        origin_scores = mixed_scores[:b//2]  # batch * patches_num * embedding_dim
        isom_scores = mixed_scores[b//2:]    # batch * patches_num * embedding_dim
        
        # sampling
        sample_set1 = set(random.sample(range(num), k = int(num * self.sample_ratio)))
        sample_set2 = set(random.sample((set(range(num)) - sample_set1), k = int(num * self.sample_ratio)))
        sample_set1 = list(sample_set1)
        sample_set2 = list(sample_set2)

        sim_origin_scores = origin_scores[:,sample_set1]
        sim_isom_scores = isom_scores[:,sample_set1]
        diff_origin_scores = origin_scores[:,sample_set2]
        diff_isom_scores = isom_scores[:,sample_set2]

        # matrix: (batch * sampled_num) * dim
        sim_origin_scores = sim_origin_scores.reshape(-1,dim)  
        sim_isom_scores = sim_isom_scores.reshape(-1,dim)
        diff_origin_scores = diff_origin_scores.reshape(-1,dim)
        diff_isom_scores = diff_isom_scores.reshape(-1,dim)

        # (batch * sampled_num) * 1
        sim = self.innner_cri(sim_origin_scores,sim_isom_scores).sum(dim = -1,keepdim = True)
        ori_diff = self.innner_cri(sim_origin_scores,diff_origin_scores).sum(dim = -1,keepdim = True)
        isom_diff = self.innner_cri(sim_isom_scores,diff_isom_scores).sum(dim = -1,keepdim = True)
        pairs_distance = sim + (self.prob_margin ** 2 * dim) - 0.5 * (ori_diff + isom_diff)

        # (batch * sampled_num) * 2
        pairs_loss = torch.max(
            torch.cat((torch.zeros_like(pairs_distance),pairs_distance),dim = 1),dim = 1
        )[0]
        
        loss = pairs_loss.mean()

        return loss

if __name__ == "__main__":

    logits = torch.rand(size = (4,10,4),dtype = torch.float32)
    probs = torch.sigmoid(logits)
    
    fda_cri = FDALoss()
    loss = fda_cri(probs)
    print(loss)

    






