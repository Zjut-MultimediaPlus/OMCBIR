import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image


class Cudata(Dataset):

    def __init__(self):
        super(Cudata, self).__init__()
        self.data_dir = r'/Users/qing/Desktop/7datasets/SUN397/100searchDataresize128/2sun397_searchDB_original_All/'
        self.datalist = os.listdir(self.data_dir)
        self.cu_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(128),
            transforms.CenterCrop(128),
        ])

    def __getitem__(self, index):
        id = self.datalist[index]
        img_path = os.path.join(self.data_dir, id)
        img = Image.open(img_path)
        img = self.cu_transform(img)
        return img

    def __len__(self):
        return len(self.datalist)


cu_data = Cudata()
for idx, x in enumerate(cu_data):
    print(x.shape)
    print(idx)
    imageName = cu_data.datalist[idx]
    print(imageName)
    save_image(x, '/Users/qing/Desktop/7datasets/SUN397/100searchDataresize128/3sun397_searchDB_resize128_All/retrivalDBImageSun397/{}'.format(imageName))