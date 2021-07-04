import torch,os
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

from models import lrcm as lrcm
from datahandler import cubdataset as cubdataset
from utils import utils
from loss import custom_loss

import config

random_seed = 524626546435
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

def train_model():

    """
    train the model.
    """
    # region Prepare
    # device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = cubdataset.CubDataset(
        config.DATA_ROOT,
        training = True,
        resize = config.RESIZE,
        crop = config.CROP,
        patch_num = [8,4,2],
    )
    train_loader = DataLoader(
        train_set,
        config.TRAIN_BATCH_SIZE,
        shuffle = True,
        num_workers = config.NUM_WORKERS,
        drop_last = True
    )
    # endregion

    # region Model
    # model.
    bb_name = "densenet161"
    
    model = lrcm.LRCM(
        bb_name = bb_name,
        backbone_pretrained_path = config.PRETRAINED_MODELS[bb_name],
        in_channels_list = [768,2112,2208],
        embedding_dim = 768,

        rlow_pool_size = [7,7,0],
        rmid_pool_size = [7,7,0],
        rhig_pool_size = [7,7,0],

        vlow_pool_size = [4,4,0],
        vmid_pool_size = [4,4,0],
        vhig_pool_size = [4,4,1],

        low_patch_num = 14 * 14,
        mid_patch_num = 7 * 7,
        hig_patch_num = 4 * 4,
        n_head = 8,
        reduced_dim = 768,
        atte_hidden_unit = 2048,
        dropout = 0.1,
        num_class = config.NUM_CLASS,
        pretrained_model_path = None,
    )

    model = model.to(device)
    print("============================================================================================================")
    print(model)
    print("============================================================================================================")
    
    # endregion
    
    # region Optim
    # criterion.
    cls_cri = nn.CrossEntropyLoss()
    fda_cri = custom_loss.FDALoss()

    # optimizer.
    optimizer = optim.SGD([
        {'params': model.r_module.stem.parameters(), 'lr': config.LR * 0.1},
        {'params': model.r_module.apdx.parameters(), 'lr': config.LR},
        {'params': model.v_module.parameters(), 'lr': config.LR},
        {'params': model.c_module.parameters(), 'lr': config.LR}
    ], momentum = config.MOMENTUM,weight_decay = config.WD)
    # endregion

    # region Scheduler
    # lr
    epoch_iters = len(train_loader)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = (config.TOTAL_EPOCH - config.WARM_UP_EPOCH) * epoch_iters,eta_min = 0)
    wrap_scheduler = utils.WarmupWrapper(optimizer,0,config.WARM_UP_EPOCH * epoch_iters,lr_scheduler)
    # endregion
    
    # region Run
    global_step = 0
    best_fine_epoch = 0
    best_valid_fine_acc = 0
    best_ens_epoch = 0
    best_valid_ens_acc = 0

    optimizer.zero_grad()
    optimizer.step()
    wrap_scheduler.step()
    
    for epoch in range(config.TOTAL_EPOCH):

        # statistics
        batch_low_cls_loss = 0
        batch_mid_cls_loss = 0
        batch_hig_cls_loss = 0
        batch_coa_cls_loss = 0
        batch_com_cls_loss = 0
        batch_fda_loss = 0

        batch_low_acc = 0
        batch_mid_acc = 0
        batch_hig_acc = 0
        batch_coa_acc = 0
        batch_com_acc = 0 

        epoch_low_acc_num = 0
        epoch_mid_acc_num = 0
        epoch_hig_acc_num = 0
        epoch_coa_acc_num = 0
        epoch_com_acc_num = 0
        epoch_train_num = 0

        model.train()

        for data in train_loader:
            
            global_step += 1

            # region Forward and Backward
            # data
            ori_img,low_img,mid_img,hig_img,low_patch_indices,mid_patch_indices,hig_patch_indices,targets = data
            low_indices = torch.stack(low_patch_indices,dim = 1).to(device)
            mid_indices = torch.stack(mid_patch_indices,dim = 1).to(device)
            hig_indices = torch.stack(hig_patch_indices,dim = 1).to(device)
            b = targets.size()[0]

            # region Coarse
            optimizer.zero_grad()
            imgs = torch.cat((ori_img,hig_img,mid_img,low_img),dim = 0).to(device)
            low_logits,mid_logits,hig_logits,low_fda_output,mid_fda_output,hig_fda_output = model(
                imgs,
                low_indices,
                mid_indices,
                hig_indices,
                step = "coarse"
            )

            # cls
            low_targets = torch.cat((targets,targets,targets,targets),dim = 0).to(device)
            low_loss = cls_cri(low_logits,low_targets)

            mid_targets = torch.cat((targets,targets,targets),dim = 0).to(device)
            mid_loss = cls_cri(mid_logits,mid_targets)

            hig_targets = torch.cat((targets,targets),dim = 0).to(device)
            hig_loss = cls_cri(hig_logits,hig_targets)

            cls_loss = low_loss + mid_loss + hig_loss

            batch_low_cls_loss += low_loss.item()
            batch_mid_cls_loss += mid_loss.item()
            batch_hig_cls_loss += hig_loss.item()

            # fda
            lambda_weights = 0.01
            fda_loss = lambda_weights * (fda_cri(low_fda_output) + fda_cri(mid_fda_output) + fda_cri(hig_fda_output))
            batch_fda_loss += fda_loss.item()

            loss = cls_loss + fda_loss
            loss.backward()
            optimizer.step()

            # evaluation
            total_logits = torch.cat((low_logits,mid_logits,hig_logits),dim = 0)
            total_probs = torch.softmax(total_logits,dim = -1)
            total_preds = torch.argmax(total_probs,dim = -1)

            low_preds,mid_preds,hig_preds = torch.split(total_preds,[b * 4,b * 3,b * 2],dim = 0)

            low_acc_num = torch.sum(low_preds == low_targets).item()
            mid_acc_num = torch.sum(mid_preds == mid_targets).item()
            hig_acc_num = torch.sum(hig_preds == hig_targets).item()

            del imgs,low_img,mid_img,hig_img
            # endregion

            # region Fine
            optimizer.zero_grad()
            ori_img = ori_img.to(device)
            coa_logits,com_logits = model(
                ori_img,
                step = "fine"
            )

            targets = targets.to(device)
            coa_loss = cls_cri(coa_logits, targets)
            com_loss = cls_cri(com_logits, targets)

            batch_coa_cls_loss += coa_loss.item()
            batch_com_cls_loss += com_loss.item()

            loss = coa_loss + com_loss
            loss.backward()
            optimizer.step()
            wrap_scheduler.step()

            coa_probs = torch.softmax(coa_logits,dim = -1)
            com_probs = torch.softmax(com_logits,dim = -1)
            coa_preds = torch.argmax(coa_probs,dim = -1)
            com_preds = torch.argmax(com_probs,dim = -1)
            coa_acc_num = torch.sum(coa_preds == targets).item()
            com_acc_num = torch.sum(com_preds == targets).item()
            # endregion
            # endregion

            # statistic
            batch_low_acc += (low_acc_num / (b * 4))
            batch_mid_acc += (mid_acc_num / (b * 3))
            batch_hig_acc += (hig_acc_num / (b * 2))
            batch_coa_acc += (coa_acc_num / b)
            batch_com_acc += (com_acc_num / b)
            
            epoch_low_acc_num += low_acc_num
            epoch_mid_acc_num += mid_acc_num
            epoch_hig_acc_num += hig_acc_num
            epoch_coa_acc_num += coa_acc_num
            epoch_com_acc_num += com_acc_num

            epoch_train_num += b

            # display
            if global_step % config.RECORD_FREQ == 0:
                
                avg_low_cls_loss = batch_low_cls_loss / config.RECORD_FREQ
                avg_mid_cls_loss = batch_mid_cls_loss / config.RECORD_FREQ
                avg_hig_cls_loss = batch_hig_cls_loss / config.RECORD_FREQ
                avg_coa_cls_loss = batch_coa_cls_loss / config.RECORD_FREQ
                avg_com_cls_loss = batch_com_cls_loss / config.RECORD_FREQ
                avg_fda_loss = batch_fda_loss / config.RECORD_FREQ

                avg_low_acc = batch_low_acc / config.RECORD_FREQ * 100
                avg_mid_acc = batch_mid_acc / config.RECORD_FREQ * 100
                avg_hig_acc = batch_hig_acc / config.RECORD_FREQ * 100
                avg_coa_acc = batch_coa_acc / config.RECORD_FREQ * 100
                avg_com_acc = batch_com_acc / config.RECORD_FREQ * 100
                
                print("E:%d-S:%d-[L:%.4f-M:%.4f-H:%.4f-CO:%.4f-CM:%.4f-F:%.4f]-[L:%.2f%%-M:%.2f%%-H:%.2f%%-CO:%.2f%%-CM:%.2f%%]" % 
                    (
                        epoch,
                        global_step,
                        avg_low_cls_loss,
                        avg_mid_cls_loss,
                        avg_hig_cls_loss,
                        avg_coa_cls_loss,
                        avg_com_cls_loss,
                        avg_fda_loss,
                        avg_low_acc,
                        avg_mid_acc,
                        avg_hig_acc,
                        avg_coa_acc,
                        avg_com_acc
                    )
                )

                with open(config.RECORD_DIR + config.SAVE_MODEL_NAME + "-train.txt","a") as f:
                    f.write("E:%d-S:%d-[L:%.4f-M:%.4f-H:%.4f-CO:%.4f-CM:%.4f-F:%.4f]-[L:%.2f%%-M:%.2f%%-H:%.2f%%-CO:%.2f%%-CM:%.2f%%]\n" % 
                    (
                        epoch,
                        global_step,
                        avg_low_cls_loss,
                        avg_mid_cls_loss,
                        avg_hig_cls_loss,
                        avg_coa_cls_loss,
                        avg_com_cls_loss,
                        avg_fda_loss,
                        avg_low_acc,
                        avg_mid_acc,
                        avg_hig_acc,
                        avg_coa_acc,
                        avg_com_acc
                    )
                )
                    batch_low_cls_loss = 0
                    batch_mid_cls_loss = 0
                    batch_hig_cls_loss = 0
                    batch_coa_cls_loss = 0
                    batch_com_cls_loss = 0
                    batch_fda_loss = 0

                    batch_low_acc = 0
                    batch_mid_acc = 0
                    batch_hig_acc = 0
                    batch_coa_acc = 0
                    batch_com_acc = 0
        
        # log
        low_cls_loss,mid_cls_loss,hig_cls_loss,coa_cls_loss,com_cls_loss,\
        low_acc,mid_acc,hig_acc,coa_acc,com_acc,fine_acc,ens_acc = valid_model(
            model,device,cls_cri
        )
        if fine_acc >= best_valid_fine_acc:
            best_valid_fine_acc = fine_acc
            best_fine_epoch = epoch
            saved_path = os.path.join(config.CHECKPOINT_SAVED_FOLDER,config.SAVE_MODEL_NAME+"-fine.pth")
            torch.save(model.state_dict(),saved_path,_use_new_zipfile_serialization = False)
        if ens_acc >= best_valid_ens_acc:
            best_valid_ens_acc = ens_acc
            best_ens_epoch = epoch
            saved_path = os.path.join(config.CHECKPOINT_SAVED_FOLDER,config.SAVE_MODEL_NAME+"-ens.pth")
            torch.save(model.state_dict(),saved_path,_use_new_zipfile_serialization = False)
        print("-" * 80)
        print("E:%d-S:%d-[L:%.4f-M:%.4f-H:%.4f-CO:%.4f-CM:%.4f]-[(Va)L:%.2f%%-M:%.2f%%-H:%.2f%%-CO:%.2f%%-CM:%.2f%%-ENS:%.2f%%]-[(Ta)L:%.2f%%-M:%.2f%%-H:%.2f%%-CO:%.2f%%-CM:%.2f%%]-[E:%d-Fine:%.2f%%]-[E:%d-ENS:%.2f%%]" % 
                (
                    epoch,
                    global_step,     
                    low_cls_loss,
                    mid_cls_loss,
                    hig_cls_loss,
                    coa_cls_loss,
                    com_cls_loss,
                    low_acc,
                    mid_acc,
                    hig_acc,
                    coa_acc,
                    com_acc,
                    ens_acc,
                    epoch_low_acc_num / (epoch_train_num * 4) * 100,
                    epoch_mid_acc_num / (epoch_train_num * 3) * 100,
                    epoch_hig_acc_num / (epoch_train_num * 2) * 100,
                    epoch_coa_acc_num / epoch_train_num * 100,
                    epoch_com_acc_num / epoch_train_num * 100,
                    best_fine_epoch,
                    best_valid_fine_acc,
                    best_ens_epoch,
                    best_valid_ens_acc,
            )
        )
        print("-" * 80)
        with open(config.RECORD_DIR + config.SAVE_MODEL_NAME + "-test.txt","a") as f:
            f.write("E:%d-S:%d-[L:%.4f-M:%.4f-H:%.4f-CO:%.4f-CM:%.4f]-[(Va)L:%.2f%%-M:%.2f%%-H:%.2f%%-CO:%.2f%%-CM:%.2f%%-ENS:%.2f%%]-[(Ta)L:%.2f%%-M:%.2f%%-H:%.2f%%-CO:%.2f%%-CM:%.2f%%]-[E:%d-Fine:%.2f%%]-[E:%d-ENS:%.2f%%]\n" % 
                (
                    epoch,
                    global_step,     
                    low_cls_loss,
                    mid_cls_loss,
                    hig_cls_loss,
                    coa_cls_loss,
                    com_cls_loss,
                    low_acc,
                    mid_acc,
                    hig_acc,
                    coa_acc,
                    com_acc,
                    ens_acc,
                    epoch_low_acc_num / (epoch_train_num * 4) * 100,
                    epoch_mid_acc_num / (epoch_train_num * 3) * 100,
                    epoch_hig_acc_num / (epoch_train_num * 2) * 100,
                    epoch_coa_acc_num / epoch_train_num * 100,
                    epoch_com_acc_num / epoch_train_num * 100,
                    best_fine_epoch,
                    best_valid_fine_acc,
                    best_ens_epoch,
                    best_valid_ens_acc,
            )
        )
    # endregion
    
def valid_model(model,device,cls_cri):
    """
    valid or test the model.
    """
    # test datasets
    test_set = cubdataset.CubDataset(config.DATA_ROOT,training = False,resize = config.RESIZE,crop = config.CROP)
    test_loader = DataLoader(test_set,1,shuffle = False,num_workers = 1)

    # valid run.
    low_cls_loss = 0
    mid_cls_loss = 0
    hig_cls_loss = 0
    coa_cls_loss = 0
    com_cls_loss = 0

    low_acc = 0
    mid_acc = 0
    hig_acc = 0
    coa_acc = 0
    com_acc = 0
    fine_acc = 0
    ens_acc = 0
    total_num = len(test_loader)

    # change to eval mode.
    model.eval()
    with torch.no_grad():
        for data in tqdm.tqdm(test_loader,desc = "valid"):
            # data
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)

            # forward
            low_logits,mid_logits,hig_logits,coa_logits,com_logits = model(imgs)
            low_loss,mid_loss,hig_loss,coa_loss,com_loss = \
                cls_cri(low_logits,targets),cls_cri(mid_logits,targets),\
                cls_cri(hig_logits,targets),cls_cri(coa_logits,targets),cls_cri(com_logits,targets)

            low_cls_loss += low_loss.item()
            mid_cls_loss += mid_loss.item()
            hig_cls_loss += hig_loss.item()
            coa_cls_loss += coa_loss.item()
            com_cls_loss += com_loss.item()

            # evaluation.
            b = targets.size()[0]
            fine_logits = coa_logits + com_logits
            ens_logits = low_logits + mid_logits + hig_logits + coa_logits + com_logits
            total_logits = torch.cat((low_logits,mid_logits,hig_logits,coa_logits,com_logits,fine_logits,ens_logits),dim = 0)
            total_probs = torch.softmax(total_logits,dim = -1)
            total_preds = torch.argmax(total_probs,dim = -1)

            low_preds,mid_preds,hig_preds,coa_preds,com_preds,fine_preds,ens_preds = torch.split(total_preds,[b,b,b,b,b,b,b],dim = 0)
            low_acc += torch.sum(low_preds == targets).item()
            mid_acc += torch.sum(mid_preds == targets).item()
            hig_acc += torch.sum(hig_preds == targets).item()
            coa_acc += torch.sum(coa_preds == targets).item()
            com_acc += torch.sum(com_preds == targets).item()
            fine_acc += torch.sum(fine_preds == targets).item()
            ens_acc += torch.sum(ens_preds == targets).item()

    return  low_cls_loss/total_num,mid_cls_loss/total_num,hig_cls_loss/total_num,\
            coa_cls_loss/total_num,com_cls_loss/total_num,\
            low_acc/total_num * 100,mid_acc/total_num * 100,hig_acc/total_num * 100,\
            coa_acc/total_num * 100,com_acc/total_num * 100,fine_acc/total_num * 100,ens_acc/total_num * 100

if __name__ == "__main__":

    # training
    train_model()