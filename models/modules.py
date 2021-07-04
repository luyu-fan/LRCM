import torch,sys
import torch.nn as nn
from models.transformer import ComprehensionLayer
from models.basic_block import Conv2dBA,Extactor, GELU

class FcNet(nn.Module):

    def __init__(
        self,
        in_features,
        hidden_units,
        num_class,
        eps = 1e-8
        ):
        
        super(FcNet,self).__init__()
        
        self.input_bn = nn.BatchNorm1d(in_features,eps = eps)
        self.fc1 = nn.Linear(in_features,hidden_units,bias = False)
        self.fc_bn = nn.BatchNorm1d(hidden_units,eps = eps)
        self.act = GELU()
        self.fc2 = nn.Linear(hidden_units,num_class,bias = False)
        
    def forward(self,x):
        
        x = self.input_bn(x)
        x = self.fc1(x)
        x = self.fc_bn(x)
        x = self.act(x)
        x = self.fc2(x)

        return x

class FdaNet(nn.Module):

    def __init__(
        self,
        pool_size,
        embedding_dim,
        hidden_units,
        latent_features,
        eps = 1e-8
        ):
        super(FdaNet,self).__init__()

        self.max_pool = nn.MaxPool2d(*pool_size)

        self.input_ln = nn.LayerNorm(embedding_dim, eps = eps)
        self.fc1 = nn.Linear(embedding_dim,hidden_units,bias = False)
        self.fc_ln = nn.LayerNorm(hidden_units, eps = eps)
        self.act = GELU()
        self.fc2 = nn.Linear(hidden_units,latent_features,bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,mixed_features,patch_ref_indices):

        mixed_features = self.max_pool(mixed_features)
        b,c,h,w = mixed_features.size()
        mixed_features = mixed_features.permute(0,2,3,1)
        mixed_features = mixed_features.reshape(b,h * w,c)

        ori_features,isom_features = torch.split(mixed_features,[b // 2, b // 2],dim = 0)
        ori_features = torch.stack(
            [ori_features[i,patch_ref_indices[i]] for i in range(b // 2)]
        )

        x = torch.cat((ori_features,isom_features),dim = 0)

        x = self.input_ln(x)
        x = self.fc1(x)
        x = self.fc_ln(x)
        x = self.act(x)
        x = self.fc2(x)
        
        x = self.sigmoid(x)
        
        return x

class RepresentationModule(nn.Module):

    def __init__(
        self,
        bb_name, 
        bb_weight_path, 
        channels_list,
        rlow_pool_size,
        rmid_pool_size,
        rhig_pool_size,
        embedding_dim,
        atte_hidden_unit,
        latent_features,
        num_class
    ):
        super(RepresentationModule,self).__init__()

        # stem
        self.stem = Extactor(bb_name,bb_weight_path)

        # conv
        fcnv = nn.ModuleList([
            Conv2dBA(channels_list[0],embedding_dim,kernel_size = 1,stride = 1,padding = 0,bn = True,act = True),
            Conv2dBA(channels_list[1],embedding_dim,kernel_size = 1,stride = 1,padding = 0,bn = True,act = True),
            Conv2dBA(channels_list[2],embedding_dim,kernel_size = 1,stride = 1,padding = 0,bn = True,act = True)
        ])

        # ccls
        ccls = nn.ModuleList([
            FcNet(embedding_dim,embedding_dim,num_class),
            FcNet(embedding_dim,embedding_dim,num_class),
            FcNet(embedding_dim,embedding_dim,num_class)
        ])

        # fdam
        fdam = nn.ModuleList([
            FdaNet(rlow_pool_size,embedding_dim,atte_hidden_unit,latent_features),
            FdaNet(rmid_pool_size,embedding_dim,atte_hidden_unit,latent_features),
            FdaNet(rhig_pool_size,embedding_dim,atte_hidden_unit,latent_features)
        ])

        # apdx
        self.apdx = nn.ModuleDict(
            {
                "fcnv":fcnv,
                "ccls":ccls,
                "fdam":fdam
            }
        )

        self.global_max_pool = nn.AdaptiveMaxPool2d((1,1))

    def forward(self,img,low_patch_indices = None,mid_patch_indices = None,hig_patch_indices = None,step = "final"):

        low_fea,mid_fea,hig_fea = self.stem(img)
        low_fda_output,mid_fda_output,hig_fda_output = None,None,None

        if step == "coarse":

            assert low_fea.size()[0] % 4 == 0
            b = low_fea.size()[0] // 4

            low_fea = self.apdx["fcnv"][0](low_fea)
            mid_fea = self.apdx["fcnv"][1](mid_fea[:3 * b])
            hig_fea = self.apdx["fcnv"][2](hig_fea[:2 * b])

            low_coarse_vectors = torch.flatten(self.global_max_pool(low_fea), 1)
            mid_coarse_vectors = torch.flatten(self.global_max_pool(mid_fea), 1)
            hig_coarse_vectors = torch.flatten(self.global_max_pool(hig_fea), 1)

            low_logits = self.apdx["ccls"][0](low_coarse_vectors)
            mid_logits = self.apdx["ccls"][1](mid_coarse_vectors)
            hig_logits = self.apdx["ccls"][2](hig_coarse_vectors)

            low_mix_features = torch.cat((low_fea[:b],low_fea[-b:]),dim = 0)
            low_fda_output = self.apdx["fdam"][0](low_mix_features,low_patch_indices)

            mid_mix_features = torch.cat((mid_fea[:b],mid_fea[-b:]),dim = 0)
            mid_fda_output = self.apdx["fdam"][1](mid_mix_features,mid_patch_indices)

            hig_mix_features = hig_fea
            hig_fda_output = self.apdx["fdam"][2](hig_mix_features,hig_patch_indices)

            return low_logits,mid_logits,hig_logits,low_fda_output,mid_fda_output,hig_fda_output

        else:
            low_fea = self.apdx["fcnv"][0](low_fea)
            mid_fea = self.apdx["fcnv"][1](mid_fea)
            hig_fea = self.apdx["fcnv"][2](hig_fea)

            low_coarse_vectors = torch.flatten(self.global_max_pool(low_fea), 1)
            mid_coarse_vectors = torch.flatten(self.global_max_pool(mid_fea), 1)
            hig_coarse_vectors = torch.flatten(self.global_max_pool(hig_fea), 1)

            if step == "fine":
                return low_fea,mid_fea,hig_fea,low_coarse_vectors,mid_coarse_vectors,hig_coarse_vectors
            else:
                low_logits = self.apdx["ccls"][0](low_coarse_vectors)
                mid_logits = self.apdx["ccls"][1](mid_coarse_vectors)
                hig_logits = self.apdx["ccls"][2](hig_coarse_vectors)
                return low_fea,mid_fea,hig_fea,low_coarse_vectors,mid_coarse_vectors,hig_coarse_vectors,low_logits,mid_logits,hig_logits

class VectorsEmbeddingModule(nn.Module):

    def __init__(
        self,
        low_pool_size,
        mid_pool_size,
        hig_pool_size,
        low_patch_num,
        mid_patch_num,
        hig_patch_num,
        embedding_dim,
        dropout = 0.0
    ):

        super(VectorsEmbeddingModule,self).__init__()
        self.low_granu_pool = nn.MaxPool2d(*low_pool_size)
        self.mid_granu_pool = nn.MaxPool2d(*mid_pool_size)
        self.hig_granu_pool = nn.MaxPool2d(*hig_pool_size)
        
        # position embeddings
        self.low_pos_embeddings = nn.Parameter(torch.rand(size = (low_patch_num + 1, embedding_dim),requires_grad = True))
        self.mid_pos_embeddings = nn.Parameter(torch.rand(size = (mid_patch_num + 1, embedding_dim),requires_grad = True))
        self.hig_pos_embeddings = nn.Parameter(torch.rand(size = (hig_patch_num + 1, embedding_dim),requires_grad = True))
        
        self.dropout = nn.Dropout2d(dropout)
     
    def forward(self,low_features,mid_features,hig_features,low_coarse_vectors,mid_coarse_vectors,hig_coarse_vectors):

        b = low_features.size()[0]
        
        # pool
        low_features = self.low_granu_pool(low_features)
        mid_features = self.mid_granu_pool(mid_features)
        hig_features = self.hig_granu_pool(hig_features)
        
        coarse_vectors = torch.cat((low_coarse_vectors,mid_coarse_vectors,hig_coarse_vectors),dim = 1)

        low_token = low_coarse_vectors.unsqueeze(1)
        mid_token = mid_coarse_vectors.unsqueeze(1)
        hig_token = hig_coarse_vectors.unsqueeze(1)
        
        # region Flatten
        b,c,h,w = low_features.size()
        low_num = h * w
        low_features = low_features.permute(0,2,3,1)
        low_features = low_features.reshape(b,low_num,c)
        low_features = torch.cat((low_token, low_features),dim = 1)
        low_features = low_features + self.low_pos_embeddings

        b,c,h,w = mid_features.size()
        mid_num = h * w
        mid_features = mid_features.permute(0,2,3,1)
        mid_features = mid_features.reshape(b,mid_num,c)
        mid_features = torch.cat((mid_token, mid_features),dim = 1)
        mid_features = mid_features + self.mid_pos_embeddings

        b,c,h,w = hig_features.size()
        hig_num = h * w
        hig_features = hig_features.permute(0,2,3,1)
        hig_features = hig_features.reshape(b,hig_num,c)
        hig_features = torch.cat((hig_token, hig_features),dim = 1)
        hig_features = hig_features + self.hig_pos_embeddings
        # endregion
        
        # dropout
        low_features,mid_features,hig_features = torch.split(
            self.dropout(
                torch.cat((low_features,mid_features,hig_features),dim = 1)
            ),
            [low_num + 1, mid_num + 1,hig_num + 1],
            dim = 1
        )
        
        return low_features,mid_features,hig_features,coarse_vectors

class ComprehensionModule(nn.Module):

    def __init__(
        self,
        embedding_dim,
        reduced_dim,
        n_head,
        atte_hidden_unit,
        dropout,
        num_class = 200
    ):
        
        super(ComprehensionModule,self).__init__()

        self.comprehension_layer = ComprehensionLayer(embedding_dim,reduced_dim,n_head,atte_hidden_unit,dropout,has_ffn = True)
        
        # double heads
        self.coarse_fc = FcNet(embedding_dim * 3,embedding_dim,num_class)
        self.com_fc = FcNet(embedding_dim * 3,embedding_dim,num_class)

    def forward(self,low_vectors,mid_vectors,hig_vectors,coarse_vectors):
        
        output1,output2,output3 = self.comprehension_layer(low_vectors,mid_vectors,hig_vectors)

        low_vectors,mid_vectors,hig_vectors = output1
        low_weights,mid_weights,hig_weights = output2
        mid_low_weights,hig_mid_weights,hig_low_weights = output3
        
        com_vectors = torch.cat(
            [
                low_vectors[:,0],
                mid_vectors[:,0],
                hig_vectors[:,0]
            ], 
            dim = 1
        )
        
        coarse_logits = self.coarse_fc(
            coarse_vectors
        )
        com_logits = self.com_fc(
           com_vectors
        )
        
        return coarse_logits,com_logits
