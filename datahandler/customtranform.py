import numpy as np,random
import PIL.Image as Image,PIL.ImageEnhance as ImageEnhance
        
class RandomRotaion(object):
    
    def __init__(self,p = 0.5):

        self.degrees = [-90,90,180,-180,270,-270]
        self.p = p

    def __call__(self,img):
        """
        rotate the image in a random degree
        """
        p = random.random()
        if p <= self.p:
            degree = random.choice(self.degrees)
            img = img.rotate(degree)
        return img

class ColorAdjustion(object):
    """
    color saturation adjustion: [0.5,1.5]
    """
    def __init__(self,p):
        self.p = p

    def __call__(self,img):
        p = random.random()
        if p <= self.p:
            fac = random.random() + 0.5
            enhance_img = ImageEnhance.Color(img)
            img = enhance_img.enhance(fac)
        return img

class ContrastAdjustion(object):
    """
    image contrast adjustion: [0.5,1.5]
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        p = random.random()
        if p <= self.p:
            fac = random.random() + 0.5
            enhance_img = ImageEnhance.Contrast(img)
            img = enhance_img.enhance(fac)
        return img

class BrightnessAdjustion(object):
    """
    image brightness adjustion: [0.5,1.5]
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        p = random.random()
        if p <= self.p:
            fac = random.random() + 0.5
            enhance_img = ImageEnhance.Brightness(img)
            img = enhance_img.enhance(fac)
        return img

class SharpnessAdjustion(object):
    """
    image sharpness adjustion: [0.5,1.5]
    """
    def __init__(self,p):
        self.p = p

    def __call__(self,img):
        p = random.random()
        if p <= self.p:
            fac = random.random() + 0.5
            enhance_img = ImageEnhance.Sharpness(img)
            img = enhance_img.enhance(fac)
        return img

class IsomTransform(object):
    
    def __init__(
        self,
        img_size = 448,           # image size
        rotp = 0.5,               # rotation degree
        colorp = 0.5,             # color adjustion fraction
        contp = 0.5,              # contrast adjustion
        brip = 0.5,               # brightness adjustion
        sharpp = 0.5,             # sharpness adjustion
        patch_num = 8             # patch_num
        ):

        super(IsomTransform,self).__init__()

        self.img_size = img_size

        self.rotation = RandomRotaion(p = rotp)
        self.color = ColorAdjustion(p = colorp)
        self.contrast = ContrastAdjustion(p = contp)
        self.brightness = BrightnessAdjustion(p = brip)
        self.sharpness = SharpnessAdjustion(p = sharpp)

        self.patch_num = patch_num

    def __call__(self,image):
        """
        get the isomorphism copy.

        Args:
            image: PIL Image

        Return:
            isomorphism image: PIL Image
        """

        assert self.img_size % self.patch_num == 0

        patch_size = self.img_size // self.patch_num

        # patches position list
        patches_pos_list = [(i + 1,j + 1) for i in range(self.patch_num) for j in range(self.patch_num)]
        random.shuffle(patches_pos_list)
        patches_ref_index = []

        image_arr = np.array(image,dtype = np.uint8)
        isomo_arr = np.zeros_like(image_arr)
        for i in range(self.patch_num):          # row
            for j in range(self.patch_num):      # col
                patch_pos = patches_pos_list[i * self.patch_num + j]
                patches_ref_index.append((patch_pos[0] - 1) * self.patch_num + (patch_pos[1] - 1))
                patch = image_arr[
                    (patch_pos[0] - 1) * patch_size:patch_pos[0] * patch_size,
                    (patch_pos[1] - 1) * patch_size:patch_pos[1] * patch_size,
                    :
                ].copy()
                
                # rotation
                patch = self.rotation(Image.fromarray(patch))
                patch = self.color(patch)
                patch = self.contrast(patch)
                patch = self.brightness(patch)
                patch = self.sharpness(patch)

                # noise
                # patch = self.noise(np.array(patch))
                patch = np.array(patch)
                isomo_arr[
                    i * patch_size:(i + 1) * patch_size,
                    j * patch_size:(j + 1) * patch_size,
                    :
                ] = patch

        return Image.fromarray(isomo_arr),patches_ref_index

if __name__ == "__main__":

    # isomo
    low_isom_transform = IsomTransform(img_size = 448,rotp = 0.5,colorp = 0.5,contp = 0.5,brip = 0.5,sharpp = 0.5,patch_num = 4)

    img = Image.open("./datahandler/test.png")
    img.show()
    isom_img,shuffled_index = low_isom_transform(img)
    isom_img.show()
