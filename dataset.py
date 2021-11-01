import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os

class ImageDataSet(data.Dataset):
    def __init__(self,data_root, input_size, open_Method='RGB'):
        super().__init__()
        self.data_root = data_root
        self.input_size = input_size
        self.open_Method = open_Method

        self.supervisied_imgs_A = [os.path.join(self.data_root, 'supervisied', 'A',img)
                                   for img in os.listdir(os.path.join(self.data_root, 'supervisied', 'A'))]
        self.supervisied_imgs_B = [os.path.join(self.data_root, 'supervisied', 'B', img)
                                   for img in os.listdir(os.path.join(self.data_root, 'supervisied', 'B'))]
        self.supervisied_imgs = []
        for img_A,img_B in zip(self.supervisied_imgs_A,self.supervisied_imgs_B):
            assert img_A.split('/')[-1] == img_B.split('/')[-1]
            self.supervisied_imgs.append((img_A,img_B))

        self.unsupervised_imgs = [os.path.join(self.data_root, 'unsupervisied',img)
                                  for img in os.listdir(os.path.join(self.data_root, 'unsupervisied'))]

        assert len(self.supervisied_imgs) > 0
        assert len(self.unsupervised_imgs) > 0

        if self.open_Method == 'RGB':
            self.transform = transforms.Compose([
                # transforms.Resize(int(self.input_size * 1)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif self.open_Method == 'L':
            self.transform = transforms.Compose([
                # transforms.Resize(int(self.input_size * 1)),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
            ])
        else:
            print("Unknown method to open img: {}".format(open_Method))
            assert 1 > 2

    def __getitem__(self,index):
        supervisied_img_A,supervisied_img_B = self.supervisied_imgs[index % len(self.supervisied_imgs)]
        unsupervised_img = self.unsupervised_imgs[index % len(self.unsupervised_imgs)]

        supervisied_img_A = Image.open(supervisied_img_A).convert(self.open_Method)
        supervisied_img_B = Image.open(supervisied_img_B).convert(self.open_Method)
        unsupervised_img = Image.open(unsupervised_img).convert(self.open_Method)

        supervisied_img_A = self.process_img(supervisied_img_A)
        supervisied_img_B = self.process_img(supervisied_img_B)
        unsupervised_img = self.process_img(unsupervised_img,)

        return {
            'supervisied_img_A' : supervisied_img_A,
            'supervisied_img_B' : supervisied_img_B,
            'unsupervised_img' : unsupervised_img,
        }

    def __len__(self):
        return max(len(self.supervisied_imgs),
                   len(self.unsupervised_imgs))

    def process_img(self,img):
        transform = self.transform
        tensor = transform(img)
        return tensor

class TestImageDataSet(data.Dataset):
    def __init__(self,data_root, open_Method='RGB'):
        super().__init__()
        self.data_root = os.path.join(data_root,'test')
        self.open_Method = open_Method

        self.imgs = [os.path.join(self.data_root,img) for img in os.listdir(self.data_root)]
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        assert len(self.imgs) > 0

    def __getitem__(self,index):
        img_dir = self.imgs[index % len(self.imgs)]
        img = Image.open(img_dir).convert(self.open_Method)
        img = self.transform(img)
        return {'name' : img_dir.split('/')[-1],
                'img' : img}

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    data_root = '/data/haokang/Semi-Supervisied-DataSet/sketch'
    input_size = 256

    dataSet = ImageDataSet(data_root=data_root,
                           input_size=input_size,
                           open_Method='L')

    dataloader = DataLoader(dataset=dataSet,
                            shuffle=True,
                            batch_size=4,
                            num_workers=4)

    for i,data in enumerate(dataloader):
        supervisied_img_A = data['supervisied_img_A']
        supervisied_img_B = data['supervisied_img_B']
        unsupervised_img = data['unsupervised_img']
        print(supervisied_img_A.size())
        break







