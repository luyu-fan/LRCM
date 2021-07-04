import os,PIL.Image as Image,random
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from . import customtranform

class BasicCUBDataset(Dataset):
    
    """
    Customed CUB dataset.
    """
    def __init__(self,data_root,training = True,resize = 550,crop = 448,ratio = 1):

        self.root = data_root
        self.is_train = training
        self.resize = resize
        self.crop = crop

        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))

        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        img_txt_file.close()

        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        label_txt_file.close()

        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        train_val_file.close()

        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]

        # current dataset
        if self.is_train:
            train_img_path = [os.path.join(self.root, 'images', train_file) for train_file in
                                   train_file_list]
            train_label = [x for i, x in zip(train_test_list, label_list) if i]
            self.datasets_list = [(img_path,img_label) for img_path,img_label in zip(train_img_path,train_label)]
        else:
            test_img_path = [os.path.join(self.root, 'images', test_file) for test_file in
                                  test_file_list]
            test_label = [x for i, x in zip(train_test_list, label_list) if not i]
            self.datasets_list = [(img_path,img_label) for img_path,img_label in zip(test_img_path,test_label)]
        
        random.shuffle(self.datasets_list)
        self.datasets_list = self.datasets_list[:int(len(self.datasets_list) * ratio)]

        # image train_transform
        self.train_transform = transforms.Compose(
            [
                transforms.Resize(self.resize,Image.BILINEAR),
                transforms.RandomCrop(self.crop),
                transforms.RandomHorizontalFlip(p = 0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(self.resize,Image.BILINEAR),
                transforms.CenterCrop(self.crop),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.datasets_list)

    def __getitem__(self,index):

        img = Image.open(self.datasets_list[index][0])
        target = self.datasets_list[index][1]

        if self.is_train:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = self.train_transform(img)
        else:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = self.test_transform(img)

        return img, target

class CubDataset(BasicCUBDataset):

    def __init__(
        self,
        data_root,
        training=True,
        resize = 550,
        crop = 448,
        rotp = 0.5,
        colorp = 0.5,
        contp = 0.5,
        brip = 0.5,
        sharpp = 0.5,
        patch_num = [8,4,2],
        ratio = 1
    ):
        super(CubDataset,self).__init__(data_root,training,resize,crop,ratio)

        del self.train_transform
        self.pretransform = transforms.Compose(
            [
                transforms.Resize(self.resize,Image.BILINEAR),
                transforms.RandomCrop(self.crop),
                transforms.RandomHorizontalFlip(p = 0.5),
            ]
        )

        self.low_copy = customtranform.IsomTransform(
            img_size = crop,
            rotp = rotp,
            colorp = colorp,
            contp = contp,
            brip = brip,
            sharpp = sharpp,
            patch_num = patch_num[0]
        )
        self.mid_copy = customtranform.IsomTransform(
            img_size = crop,
            rotp = rotp,
            colorp = colorp,
            contp = contp,
            brip = brip,
            sharpp = sharpp,
            patch_num = patch_num[1]
        )
        self.hig_copy = customtranform.IsomTransform(
            img_size = crop,
            rotp = rotp,
            colorp = colorp,
            contp = contp,
            brip = brip,
            sharpp = sharpp,
            patch_num = patch_num[2]
        )
        
        self.endtransform = transforms.Compose(
            [   
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self,index):

        img = Image.open(self.datasets_list[index][0])
        target = self.datasets_list[index][1]
        
        # train sample
        if self.is_train:
          
            if img.mode != "RGB":
                img = img.convert("RGB")
            ori_img = self.pretransform(img)

            # get the different granu-level copy images
            low_img,low_patch_indices = self.low_copy(ori_img)
            mid_img,mid_patch_indices = self.mid_copy(ori_img)
            hig_img,hig_patch_indices = self.hig_copy(ori_img)

            ori_img = self.endtransform(ori_img)
            low_img = self.endtransform(low_img)
            mid_img = self.endtransform(mid_img)
            hig_img = self.endtransform(hig_img)
            
            return ori_img,low_img,mid_img,hig_img,low_patch_indices,mid_patch_indices,hig_patch_indices,target
            
        # valid or test
        else:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = self.test_transform(img)
        return img, target



