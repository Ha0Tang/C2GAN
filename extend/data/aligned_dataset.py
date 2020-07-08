import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)

        self.AB_paths = sorted(make_dataset(self.dir_AB))

        # assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        # AB = Image.open(AB_path).convert('RGB')
        # ABC = Image.open(AB_path).convert('RGB')
        ABCD = Image.open(AB_path).convert('RGB')
        # imgplot = plt.imshow(ABC)
        # AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        # ABC = ABC.resize((self.opt.loadSize * 3, self.opt.loadSize), Image.BICUBIC)
        ABCD = ABCD.resize((self.opt.loadSize * 4, self.opt.loadSize), Image.BICUBIC)
        # AB = self.transform(AB)
        # ABC = self.transform(ABC)
        ABCD = self.transform(ABCD)

        # w_total = ABC.size(2)
        w_total = ABCD.size(2)
        # print('w_total', w_total)
        # w = int(w_total / 2)
        # w = int(w_total / 3)
        w = int(w_total / 4)
        # h = AB.size(1)
        # h = ABC.size(1)
        # w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        # h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))
        h = ABCD.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        # A = AB[:, h_offset:h_offset + self.opt.fineSize,
        #        w_offset:w_offset + self.opt.fineSize]
        # B = AB[:, h_offset:h_offset + self.opt.fineSize,
        #        w + w_offset:w + w_offset + self.opt.fineSize]
        # A = ABC[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        # B = ABC[:, h_offset:h_offset + self.opt.fineSize, w + w_offset:w + w_offset + self.opt.fineSize]
        # C = ABC[:, h_offset:h_offset + self.opt.fineSize, w*2 + w_offset:w*2 + w_offset + self.opt.fineSize]
        A = ABCD[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        B = ABCD[:, h_offset:h_offset + self.opt.fineSize, w + w_offset:w + w_offset + self.opt.fineSize]
        C = ABCD[:, h_offset:h_offset + self.opt.fineSize, w*2 + w_offset:w*2 + w_offset + self.opt.fineSize]
        D = ABCD[:, h_offset:h_offset + self.opt.fineSize, w*3 + w_offset:w*3 + w_offset + self.opt.fineSize]

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        # if (not self.opt.no_flip) and random.random() < 0.5:
        #     idx = [i for i in range(A.size(2) - 1, -1, -1)]
        #     idx = torch.LongTensor(idx)
        #     A = A.index_select(2, idx)
        #     B = B.index_select(2, idx)
        #     C = C.index_select(2, idx)
        #     D = D.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
    
        return {'A': A, 'B': B, 'C': C, 'D': D,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
