""" train and test dataset

author baiyu
"""
import os
import sys
import pickle
import random
import glob
import string
from collections import namedtuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import lmdb
import cv2


Records = namedtuple('Records', ['labels', 'insts'])


#def lmdb_writer(db_path, lmdb_path):
#    map_size = 10 << 40
#    #print(lmdb_path)
#    env = lmdb.open(lmdb_path, map_size=map_size)
#
#    #with self.env.begin(write=True) as txn:
#        #for fp in db_pathes:
#        #with open(fp, 'rb') as f:
#        #with open(db_path, '')
#    image = np.load(db_path[0])
#    mask = np.load(db_path[1])
#    tissue_type = np.load(db_path[2])
#    print(db_path[0])
#    print(db_path[1])
#    print(db_path[2])
#    a = set(tissue_type)
#    print(len(a), a)
#    #with open()
#    global count
#    with env.begin(write=True) as txn:
#        for image, mask, tissue_type in zip(image, mask, tissue_type):
#            count += 1
#
#            data = {
#                'image' : image,
#                'mask' : mask,
#                'type' : tissue_type
#            }
#
#            #data = pickle.loads(data)
#            txn.put(pickle.dumps(count), pickle.dumps(data))
#            print(count)
#            #ff = np.all(image != (0,0,0), axis=2)
#            #print(np.all(image != (0, 0, 0))
#            #ff = np.where(image != (0,0,0))
#            #print(image[image != (0, 0, 0, 0, 0, 0)].size)
#            #print(mask.shape, mask.max())
#            # bg != 0
#            #bg_mask = bg != 0 # non-bg
#            #print(bg)
#
#
#            #bg = mask[:, :, 5]
#            #kernel = np.ones((3,3),np.uint8)
#            #dilate = cv2.dilate(bg, kernel,iterations = 3)
#            #contour = cv2.bitwise_xor(dilate, bg)
#            #image[contour == 1, :] = (0, 0, 0)
#
#
#
#
#            #cv2.imwrite('tmp/test.jpg', image)
#            #sys.exit()
#            #print(bg_mask)
#            #print(mask[:, :, 5].shape)
#            #print(bg[bg_mask][:3].size)
#            #print()
#
#
#            #print(image_idx, image.shape, mask.shape, tissue_type)


def random_size(images, cls_ids):
    #images = []
    #cls_ids = []
    sizes = [64, 128, 224]
    size = random.choice(sizes)
    if size == 224:
        return torch.stack(images), torch.stack(cls_id)

    def crop(image):
        row = 224 - size
        col = 224 - size
        while True:
            r_idx = random.choice(range(row))
            c_idx = random.choice(range(col))

            image = image[r_idx : r_idx + size, c_idx, : c_idx + size]
            mask = np.all(image == (0, 0, 0), axis=2)
            if mask.sum() / mask.size >= 0.1:
                break

        return image

    #def crops(image):
    #    return [crop(image, size) for _ in range(num_crops)]

    res_images = []
    res_labels = []
    for image, cls_id in zip(images, cls_ids):
        #print(image.shape, cls_id)
        num_crops = 224 / size * 224 / size
        images = [crop(image) for _ in range(num_crops)]
        labels = [cls_id for _ in range(cls_id)]
        res_images.extend(images)
        res_labels.extend(labels)

    return torch.stack(res_images), torch.stack(res_labels)
#def lmdb_writer(db_path, lmdb_path):
#    map_size = 10 << 40
#    #print(lmdb_path)
#    env = lmdb.open(lmdb_path, map_size=map_size)
#
#    #with self.env.begin(write=True) as txn:
#        #for fp in db_pathes:
#        #with open(fp, 'rb') as f:
#        #with open(db_path, '')
#    image = np.load(db_path[0])
#    mask = np.load(db_path[1])
#    tissue_type = np.load(db_path[2])
#    print(db_path[0])
#    print(db_path[1])
#    print(db_path[2])
#    a = set(tissue_type)
#    print(len(a), a)
#    #with open()
#    global count
#    with env.begin(write=True) as txn:
#        for image, mask, tissue_type in zip(image, mask, tissue_type):
#            count += 1
#
#            data = {
#                'image' : image,
#                'mask' : mask,
#                'type' : tissue_type
#            }
#
#            #data = pickle.loads(data)
#            txn.put(pickle.dumps(count), pickle.dumps(data))
#            print(count)
#            #ff = np.all(image != (0,0,0), axis=2)
#            #print(np.all(image != (0, 0, 0))
#            #ff = np.where(image != (0,0,0))
#            #print(image[image != (0, 0, 0, 0, 0, 0)].size)
#            #print(mask.shape, mask.max())
#            # bg != 0
#            #bg_mask = bg != 0 # non-bg
#            #print(bg)
#
#
#            #bg = mask[:, :, 5]
#            #kernel = np.ones((3,3),np.uint8)
#            #dilate = cv2.dilate(bg, kernel,iterations = 3)
#            #contour = cv2.bitwise_xor(dilate, bg)
#            #image[contour == 1, :] = (0, 0, 0)
#
#
#
#
#            #cv2.imwrite('tmp/test.jpg', image)
#            #sys.exit()
#            #print(bg_mask)
#            #print(mask[:, :, 5].shape)
#            #print(bg[bg_mask][:3].size)
#            #print()
#
#
#            #print(image_idx, image.shape, mask.shape, tissue_type)


#class Records:
#    def __init__(self):
#        self.labels = []
#        self.insts = {}
def pad_patch(min_x, min_y, max_x, max_y, output_size):
        width = max_x - min_x
        height = max_y - min_y

        if width < output_size:
            w_diff_left = (output_size - width) // 2
            w_diff_right = output_size - width - w_diff_left
            min_x = max(0, min_x - w_diff_left)
            max_x = max_x + w_diff_right
        if height < output_size:
            h_diff_top = (output_size - height) // 2
            h_diff_bot = output_size - height - h_diff_top
            min_y = max(0, min_y - h_diff_top)
            max_y = max_y + h_diff_bot

        return min_x, min_y, max_x, max_y


def write_lmdb(image_path, lmdb_path):
    map_size = 10 << 40
    #print(lmdb_path)
    env = lmdb.open(lmdb_path, map_size=map_size)

    #with self.env.begin(write=True) as txn:
    map_size = 10 << 40
    #print(lmdb_path)

    with env.begin(write=True) as txn:
        for image_fp in glob.iglob(os.path.join(image_path, '**', '*.jpg'), recursive=True):
            image = cv2.imread(image_fp, -1)
            image = pickle.dumps(image)
            image_name = os.path.basename(image_fp)
            txn.put(image_name.encode(), image)

def crop_patches(lmdb_folder, save_folder):
    env = lmdb.open(lmdb_folder, map_size=1099511627776, readonly=True, lock=False)
    with env.begin(write=False) as txn:
        image_names = [key for key in txn.cursor().iternext(keys=True, values=False)]
        #count = 0
        for name in image_names:
            data = txn.get(name)
            #count += 1
            #if count != 444:
                #continue
            #print(name)

            #print(pickle.loads(name))
            data = pickle.loads(data)
            mask = data['mask']
            image = data['image']

            for label in range(5):
                cls_mask = mask[:, :, label]
                insts = np.unique(cls_mask)[1:]
                for inst in insts:
                    #cls_mask[cls_mask == ]
                    row, col = np.asarray(cls_mask == inst).nonzero()
                    rmin = row.min()
                    rmax = row.max()
                    cmin = col.min()
                    cmax = col.max()
                    rmin, cmin, rmax, cmax = pad_patch(rmin, cmin, rmax, cmax, 64)

                    image_cp = image.copy()
                    bg_mask = mask[:, :, 5].astype('bool')
                    inst_mask = mask[:, :, label].astype('bool')
                    final_mask = inst_mask | bg_mask
                    image_cp[~final_mask] = 0

                    #image = cv2.rectangle(image, (cmin, rmin), (cmax, rmax), (255, 0, 0), 2)

                    patch = image_cp[rmin : rmax, cmin : cmax]
                    #print(patch.shape)
                    image_name = '{}_{}_{}.jpg'.format(pickle.loads(name), label, inst)
                    cv2.imwrite(os.path.join(save_folder, image_name), image_cp)
                    #cv2.imwrite(os.path.join('tmp', image_name), image_cp)
                    #print(image_name)

                    #print(pickle.loads(name), label, inst)

            #cv2.imwrite('tmp/tt.jpg', image)
            #sys.exit()
                #print(cls_mask.shape)
                #insts = np.unique()
            #if name not in self.labels:

            # remove images without any cells
#            all_insts = np.unique(mask)[2:].tolist()
#            if not all_insts:
#                continue
#
#            labels = np.asarray(mask[:, :, :self.num_classes]).nonzero()[2]
#            labels = np.unique(labels)
#            for label in labels:
#                tmp = mask[:, :, label]
#                insts[label] = np.unique(tmp[tmp != 0])
#
                    #ids = insts[label]
                    #for i in ids:
                        #row_cen, col_cen = np.asarray(mask[:, :, label] == i).nonzero()
            #label = np.random.default_rng().choice(chan)
            #mask = mask[:, :, label]

                        #row_cen = (row_cen.max() + row_cen.min()) // 2
                        #col_cen = (col_cen.max() + col_cen.min()) // 2
                        #print(pickle.loads(name), label, row_cen, col_cen, row_cen.max() )
            #label = random.choice(chan)
                        ##if tmp[tmp == 1].size != 0:
                        #    np.save('tmp.npy', tmp)
                        #    cv2.imwrite('heihei.jpg', data['image'])
                        #    sys.exit()

                        #print(insts[label])



#from utils import compute_mean_std

class PanNukEx(Dataset):
    def __init__(self, path, image_size, transforms=None):
        self.env = lmdb.open(path, map_size=1099511627776, readonly=True, lock=False)
        self.labels = {}
        self.inst2cls = {}
        self.num_classes = 5

        class_names = ['Esophagus', 'Lung', 'Breast', 'Stomach', 'Prostate', 'Thyroid', 'Kidney', 'Ovarian', 'Adrenal_gland', 'Liver', 'Skin', 'Cervix', 'Bladder', 'Colon', 'Pancreatic', 'HeadNeck', 'Bile-duct', 'Uterus', 'Testis']

        #with self.env.begin(write=False) as txn:
        #    self.image_names= [key.decode() for key in txn.cursor().iternext(keys=True, values=False)]

        cache_file = '_cache_' + ''.join(c for c in path if c in string.ascii_letters)
        cache_path = os.path.join(path, cache_file)
        if os.path.isfile(cache_path):
            self.image_records = pickle.load(open(cache_path, "rb"))
            self.image_names = list(self.image_records.keys())
        else:
            with self.env.begin(write=False) as txn:
                self.image_names = [key for key in txn.cursor().iternext(keys=True, values=False)]
            self.image_records = self.pre_process()
            pickle.dump(self.image_records, open(cache_path, "wb"))

        self.transforms = transforms

        self.image_size = image_size
        #self.mean = (0.7862793912386359, 0.6027306811087783, 0.7336620786688793) #bgr
        self.mean = [0.73646324, 0.56556627, 0.70180897] # bgr
        #self.std = (0.2111620715800869, 0.24114924152086661, 0.23603441662670357)
        self.std = [0.18869222, 0.21968669, 0.17277594] # bgr

        #from torchvision import transforms
        #train_transforms = transforms.Compose([
        #    transforms.ToPILImage(),
        #    transforms.RandomHorizontalFlip(),
        #    transforms.RandomVerticalFlip(),
        #    #transforms.RandomRotation(25),
        #    transforms.RandomResizedCrop(64, scale=(0.9, 1.1)),
        #    transforms.RandomApply(torch.nn.ModuleList([
        #            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)
        #    ]), p=0.3),
        #    #transforms.ToTensor(),
        #    #transforms.Normalize(mean, std)
        #])
        self.transforms = transforms



    #def herizontal_flip(self, image, mask):
    #    if random.random() > 0.5:
    #         cv2.flip(image, 0)
    #         cv2.flip(mask, 0)

    #    return image, mask

    #def vertical_flip(self, image, mask):
    #    if random.random() > 0.5:
    #        cv2.flip(image, 1)
    #        cv2.flip(mask, 1)

    #    return image, mask

    def rotate_image(self, image, mask, row_cen, col_cen):
        def write(mask, idx):
            mask1 = mask.copy()
            for idx, i in enumerate(np.unique(mask1)):
                mask1[mask1 == i] = inter * (idx + 1)
            #cv2.imwrite('mask{}.png'.format(idx), mask1)

        if random.random() > 0.5:
            angle = random.randint(-45, 45)
            #image_center = tuple(np.array(image.shape[1::-1]) / 2)
            image_center = col_cen, row_cen
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
            #print(np.unique(mask), 'before')
            #inter = 255 // len(np.unique(mask))
            #write(mask, 0)
            #mask_tmp = mask.copy()
            #for idx, i in enumerate(np.unique(mask_tmp)):
            #    mask_tmp[mask_tmp == i] = inter * (idx + 1)

            mask = cv2.warpAffine(mask, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST)
            #after_mask = after_mask1.copy()
            #print(np.unique(after_mask1), 'after')
            #write(after_mask1, 1)

            #for idx, i in enumerate(np.unique(mask)):
            #    after_mask[after_mask == i] = inter * (idx + 1)
            #cv2.imwrite('mask2.png', mask)
            #mask = after_mask1
        return image, mask

    def __len__(self):
        return len(self.image_names)

    def bbox(self, row, col):
        rmin = row.min()
        rmax = row.max()
        cmin = col.min()
        cmax = col.max()

        return rmin, cmin, rmax, cmax

    def pad_patch(self, min_x, min_y, max_x, max_y, output_size):
        width = max_x - min_x
        height = max_y - min_y

        if width < output_size:
            w_diff_left = (output_size - width) // 2
            w_diff_right = output_size - width - w_diff_left
            min_x = max(0, min_x - w_diff_left)
            max_x = max_x + w_diff_right
        if height < output_size:
            h_diff_top = (output_size - height) // 2
            h_diff_bot = output_size - height - h_diff_top
            min_y = max(0, min_y - h_diff_top)
            max_y = max_y + h_diff_bot

        return min_x, min_y, max_x, max_y

    #def draw_all(self, image, mask):
    #    for i in range(5):
    #        print(mask.shape)
    #        bg = mask[:, :, i]
    #        print(bg[bg!=0].size)
    #        bg = bg != 0
    #        print(bg.shape, 111)
    #        kernel = np.ones((3,3),np.uint8)
    #        dilate = cv2.dilate(bg.astype('uint8'), kernel,iterations = 3)
    #        contour = cv2.bitwise_xor(dilate, bg.astype('uint8'))
    #        image[contour == 1, :] = (i * 50, i * 50, i * 50)

    #    cv2.imwrite('tmp/test.png', image)

    def pre_process(self):
        res = {}
        with self.env.begin(write=False) as txn:
            for name in self.image_names:
                data = txn.get(name)
                #print(name)

                #print(pickle.loads(name))
                data = pickle.loads(data)
                mask = data['mask']
                #if name not in self.labels:

                # remove images without any cells
                all_insts = np.unique(mask)[2:].tolist()
                if not all_insts:
                    continue

                labels = np.asarray(mask[:, :, :self.num_classes]).nonzero()[2]
                labels = np.unique(labels)
                insts = {}
                for label in labels:
                    tmp = mask[:, :, label]
                    insts[label] = np.unique(tmp[tmp != 0])

                    #ids = insts[label]
                    #for i in ids:
                        #row_cen, col_cen = np.asarray(mask[:, :, label] == i).nonzero()
            #label = np.random.default_rng().choice(chan)
            #mask = mask[:, :, label]

                        #row_cen = (row_cen.max() + row_cen.min()) // 2
                        #col_cen = (col_cen.max() + col_cen.min()) // 2
                        #print(pickle.loads(name), label, row_cen, col_cen, row_cen.max() )
            #label = random.choice(chan)
                        ##if tmp[tmp == 1].size != 0:
                        #    np.save('tmp.npy', tmp)
                        #    cv2.imwrite('heihei.jpg', data['image'])
                        #    sys.exit()

                        #print(insts[label])

                res[name] = Records(labels=labels, insts=insts)

        self.image_names = list(res.keys())
        return res



    #def load_data(self, idx):
    #    with self.env.begin(write=False) as txn:
    #        while True:
    #            data = txn.get(name)
    #            #print(name)

    #            #print(pickle.loads(name))
    #            data = pickle.loads(data)
    #            image = data['image']
    #            mask = data['mask']
    #            if name not in self.labels:
    #                self.labels[name] = np.unique(mask)[2:].tolist()

    #            if self.labels[name]:
    #                return image, mask

    def __getitem__(self, idx):


        name = self.image_names[idx]

        #cls_id = random.choice(range(5))
        with self.env.begin(write=False) as txn:
            data = txn.get(name)
            #print(name)

            #print(pickle.loads(name))
            data = pickle.loads(data)
            image = data['image']
            mask = data['mask']

            record = self.image_records[name]

            ### select nonzero cell type:
            #_, _, chan = np.asarray(mask[:, :, :5]).nonzero()
            label = np.random.default_rng().choice(record.labels)
            #mask = mask[:, :, label]

            ### select nonzero cell type:
            #row_cen, col_cen, chan = np.asarray(mask == inst_id).nonzero()

            #if name not in self.labels:
                #self.labels[name] = [[], [], [], [], []]

            #if not self.labels[name][label]:
                #self.labels[name][label] = np.unique(mask).tolist()

                #self.labels[name] = insts
                #if len(insts)== 0:
                #    print(pickle.loads(name), idx)
                #    print(np.unique(mask))
                #    cv2.imwrite('tmp/test.jpg', image)
                #    sys.exit()
                #    return image, mask
                #print(self.labels[name], 'unique')


            #print(np.unique(mask))
            #inst_id = random.choice(self.labels[name])
            #print(record.insts[label])
            inst_id = np.random.default_rng().choice(record.insts[label])
            #print('inst_id', inst_id)
            #mask = mask == inst_id
            #cls_id = np.sum(mask, axis=(0, 1)).nonzero()[0].item()
            #print(inst_id not in self.inst2cls)
            #if inst_id not in self.inst2cls:
            #print(label)
            #print(np.unique(mask[:, :, label]))
            row_cen, col_cen = np.asarray(mask[:, :, label] == inst_id).nonzero()
            #label = np.random.default_rng().choice(chan)
            #mask = mask[:, :, label]

            row_cen = (row_cen.max() + row_cen.min()) // 2
            col_cen = (col_cen.max() + col_cen.min()) // 2
            #label = random.choice(chan)
                #print(row_cen, col_cen)
                #print(chan, inst_id)
                #if not np.all(chan == chan[0]):

                    #print(idx, inst_id, chan)
                    #np.save('test.npy', mask)
                    #cv2.imwrite('heihei.jpg', image)

                #assert np.all(chan == chan[0])
                #self.labels[inst_id] = (row_cen, col_cen, chan[0])
                #print(self.labels[inst_id][2], 44444)

            #row_cen, col_cen, label = self.labels[inst_id]
            cell_mask = mask[:, :, label].copy()
           #print(np.unique(cell_mask), 'cell_mask')
            cell_mask[cell_mask !=0] = 1
            bg_mask = mask[:, :, 5]
            image_mask = cell_mask.astype('bool') | bg_mask.astype('bool')
            image[~image_mask] = 0
            inst_mask = mask[:, :, label]
            #print(np.unique(inst_mask), 'inst_mask')
            image, inst_mask = self.rotate_image(image, inst_mask, row_cen, col_cen)


            #print(inst_mask == inst_id)
            #print(inst_id)
            #np.save('aa.npy', inst_mask)
            #cv2.imwrite('aa.jpg', image)
            row, col = np.asarray(inst_mask == inst_id).nonzero()

            box = self.bbox(row, col)
            #print(box, 11)
            #print(box, 'box')
            box = self.pad_patch(*box, self.image_size)
            #print(box, 22)
            rmin, cmin, rmax, cmax = box

            #image = cv2.rectangle(image, (cmin, rmin), (cmax, rmax), (255, 0, 0), 2)
            image = image[rmin:rmax + 1, cmin:cmax + 1]
            #print(np.sum(mask == 6, axis=2), 1111)
            #print(np.sum(mask == 6, axis=2).sum(), 1111)
            #print(np.sum(mask == 6, axis=2).nonzero())
            #self.draw_all(image, mask)
            #cls_mask = mask[:, :, cls_id]
            #cls_mask[cls_mask != 0] = 1
            #bg_mask = mask[:, :, 5]
            #mask = np.bitwise_or(bg_mask.astype('bool'), cls_mask.astype('bool'))
            #image[mask == 0] = 0
            #image.save('tmp/2233.jpg')
            if self.transforms:
                image = self.transforms(image.astype('uint8'))

            return image, label
