import torch
import torch.nn as nn

from models.modules import *

class LRCM(nn.Module):

    def __init__(
        self,
        bb_name,
        backbone_pretrained_path,
        in_channels_list,
        embedding_dim,
        rlow_pool_size,
        rmid_pool_size,
        rhig_pool_size,
        vlow_pool_size,
        vmid_pool_size,
        vhig_pool_size,
        low_patch_num,
        mid_patch_num,
        hig_patch_num,
        n_head,
        reduced_dim,
        atte_hidden_unit,
        dropout,
        num_class = 200,
        pretrained_model_path = None
    ):
        super(LRCM,self).__init__()

        latent_features = embedding_dim // 4

        self.r_module = RepresentationModule(
            bb_name = bb_name,
            bb_weight_path = backbone_pretrained_path,
            channels_list = in_channels_list,
            rlow_pool_size = rlow_pool_size,
            rmid_pool_size = rmid_pool_size,
            rhig_pool_size = rhig_pool_size,
            embedding_dim = embedding_dim,
            atte_hidden_unit = atte_hidden_unit,
            latent_features = latent_features,
            num_class = num_class
        )

        self.v_module = VectorsEmbeddingModule(
            low_pool_size = vlow_pool_size,
            mid_pool_size = vmid_pool_size,
            hig_pool_size = vhig_pool_size,
            low_patch_num = low_patch_num,
            mid_patch_num = mid_patch_num,
            hig_patch_num = hig_patch_num,
            embedding_dim = embedding_dim,
            dropout = dropout
        )

        self.c_module = ComprehensionModule(
            embedding_dim = embedding_dim,
            reduced_dim = reduced_dim,
            n_head = n_head,
            atte_hidden_unit = atte_hidden_unit,
            dropout = dropout,
            num_class = num_class
        )

        if pretrained_model_path is not None:
            self.load_state_dict(torch.load(pretrained_model_path))
    
    def forward(self,img,low_patch_indices = None,mid_patch_indices = None,hig_patch_indices = None,step = "final"):

        if step == "coarse":
            return self.r_module(img,low_patch_indices,mid_patch_indices,hig_patch_indices,step)
        elif step == "fine":
            return self.c_module(*self.v_module(*self.r_module(img,low_patch_indices,mid_patch_indices,hig_patch_indices,step)))
        else:
            low_fea,mid_fea,hig_fea,\
            low_coarse_vectors,mid_coarse_vectors,hig_coarse_vectors,\
            low_logits,mid_logits,hig_logits = self.r_module(img,low_patch_indices,mid_patch_indices,hig_patch_indices,step)
            coarse_logits,com_logits = self.c_module(*self.v_module(low_fea,mid_fea,hig_fea,low_coarse_vectors,mid_coarse_vectors,hig_coarse_vectors))
            return low_logits,mid_logits,hig_logits,coarse_logits,com_logits