from random import sample, shuffle
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import cvtColor, preprocess_input

class YoloDataset(Dataset):
    def __init__(self, annotation_lines, clean_lines, input_shape, num_classes, anchors, anchors_mask, epoch_length, train, use_dual_fog=False):
        super(YoloDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.clean_lines = clean_lines
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.anchors            = anchors
        self.anchors_mask       = anchors_mask
        self.epoch_length       = epoch_length
        self.train              = train
        self.use_dual_fog       = use_dual_fog  # Generate two fog views for contrastive learning
        self.epoch_now          = -1
        self.length             = len(self.annotation_lines)
        self.bbox_attrs         = 5 + num_classes

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        index  = index % self.length
        
        if self.use_dual_fog and self.train:
            # Generate two different fog views for contrastive learning
            image_v1, box, clearimg, image_v2 = self.get_dual_fog_data(
                self.annotation_lines[index],
                self.clean_lines[index],
                self.input_shape
            )
            image_v1    = np.transpose(preprocess_input(np.array(image_v1, dtype=np.float32)), (2, 0, 1))
            image_v2    = np.transpose(preprocess_input(np.array(image_v2, dtype=np.float32)), (2, 0, 1))
            clearimg    = np.transpose(preprocess_input(np.array(clearimg, dtype=np.float32)), (2, 0, 1))
            box         = np.array(box, dtype=np.float32)
            
            nL          = len(box)
            labels_out  = np.zeros((nL, 6))
            if nL:
                box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
                box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
                box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
                box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
                labels_out[:, 1] = box[:, -1]
                labels_out[:, 2:] = box[:, :4]
            return image_v1, image_v2, labels_out, clearimg
        else:
            # Original single image mode
            image, box, clearimg = self.get_random_data(
                self.annotation_lines[index],
                self.clean_lines[index],
                self.input_shape,
                random=self.train
            )
            image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
            box         = np.array(box, dtype=np.float32)
            clearimg    = np.transpose(preprocess_input(np.array(clearimg, dtype=np.float32)), (2, 0, 1))
            nL          = len(box)
            labels_out  = np.zeros((nL, 6))
            if nL:
                box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
                box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
                box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
                box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
                labels_out[:, 1] = box[:, -1]
                labels_out[:, 2:] = box[:, :4]
            return image, labels_out, clearimg
    
    def apply_fog_augmentation(self, image_data, fog_intensity=None):
        """
        Apply synthetic fog augmentation to simulate different fog densities.
        This creates variation for contrastive learning.
        
        Args:
            image_data: numpy array (H, W, 3)
            fog_intensity: float in [0, 1], None for random
        
        Returns:
            fogged_image: numpy array (H, W, 3)
        """
        if fog_intensity is None:
            fog_intensity = self.rand(0.1, 0.6)  # Random fog density
        
        # Simple additive fog model (can be replaced with more sophisticated methods)
        # I(x) = J(x) * t(x) + A * (1 - t(x))
        # where t(x) is transmission, A is atmospheric light
        
        # Estimate atmospheric light (simple approximation)
        A = self.rand(0.7, 0.9)
        
        # Random transmission based on fog intensity
        transmission = 1.0 - fog_intensity
        
        # Apply fog
        fogged = image_data.astype(np.float32) / 255.0
        fogged = fogged * transmission + A * (1 - transmission)
        fogged = np.clip(fogged * 255.0, 0, 255).astype(np.uint8)
        
        return fogged
    
    def get_dual_fog_data(self, annotation_line, clean_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4):
        """
        Generate two different fog views of the same image for contrastive learning.
        
        Returns:
            image_v1: first fog view
            box: bounding boxes
            clearimg: clean image
            image_v2: second fog view (different fog intensity)
        """
        line    = annotation_line.split()
        clearline = clean_line.split()
        
        # Load hazy and clean images
        image   = Image.open(line[0])
        image   = cvtColor(image)
        clearimg = Image.open(clearline[0])
        clearimg = cvtColor(clearimg)
        
        iw, ih  = image.size
        h, w    = input_shape
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
        
        # Apply same geometric augmentation to both views
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        
        image = image.resize((nw,nh), Image.BICUBIC)
        clearimg = clearimg.resize((nw, nh), Image.BICUBIC)
        
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image
        new_clearimg = Image.new('RGB', (w, h), (128, 128, 128))
        new_clearimg.paste(clearimg, (dx, dy))
        clearimg = new_clearimg
        
        flip = self.rand()<.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            clearimg = clearimg.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Convert to numpy ONCE for both views
        base_hazy = np.array(image, np.uint8)
        clear_image_data = np.array(clearimg, np.uint8)
        
        # Apply DIFFERENT fog augmentations to create two views
        # View 1: random fog intensity
        fog_intensity_v1 = self.rand(0.15, 0.5)
        image_data_v1 = self.apply_fog_augmentation(base_hazy, fog_intensity_v1)
        
        # View 2: different fog intensity (ensure meaningful difference)
        fog_intensity_v2 = self.rand(0.15, 0.5)
        while abs(fog_intensity_v2 - fog_intensity_v1) < 0.1:  # Ensure views are different
            fog_intensity_v2 = self.rand(0.15, 0.5)
        image_data_v2 = self.apply_fog_augmentation(base_hazy, fog_intensity_v2)
        
        # Apply color jittering (same for both views for consistency)
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        
        # Process view 1
        hue1, sat1, val1 = cv2.split(cv2.cvtColor(image_data_v1, cv2.COLOR_RGB2HSV))
        dtype = image_data_v1.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        image_data_v1 = cv2.merge((cv2.LUT(hue1, lut_hue), cv2.LUT(sat1, lut_sat), cv2.LUT(val1, lut_val)))
        image_data_v1 = cv2.cvtColor(image_data_v1, cv2.COLOR_HSV2RGB)
        
        # Process view 2
        hue2, sat2, val2 = cv2.split(cv2.cvtColor(image_data_v2, cv2.COLOR_RGB2HSV))
        image_data_v2 = cv2.merge((cv2.LUT(hue2, lut_hue), cv2.LUT(sat2, lut_sat), cv2.LUT(val2, lut_val)))
        image_data_v2 = cv2.cvtColor(image_data_v2, cv2.COLOR_HSV2RGB)
        
        # Process clear image
        hue_c, sat_c, val_c = cv2.split(cv2.cvtColor(clear_image_data, cv2.COLOR_RGB2HSV))
        clear_image_data = cv2.merge((cv2.LUT(hue_c, lut_hue), cv2.LUT(sat_c, lut_sat), cv2.LUT(val_c, lut_val)))
        clear_image_data = cv2.cvtColor(clear_image_data, cv2.COLOR_HSV2RGB)
        
        # Process bounding boxes
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)]
        
        return image_data_v1, box, clear_image_data, image_data_v2
    
    def get_random_data(self, annotation_line, clean_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line = annotation_line.split()
        clearline = clean_line.split()
        image = Image.open(line[0])
        image = cvtColor(image)
        clearimg = Image.open(clearline[0])
        clearimg = cvtColor(clearimg)
        iw, ih = image.size
        h, w    = input_shape
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)
            clearimg = clearimg.resize((nw, nh), Image.BICUBIC)
            new_clearimg = Image.new('RGB', (w, h), (128, 128, 128))
            new_clearimg.paste(clearimg, (dx, dy))
            clear_image_data = np.array(new_clearimg, np.float32)
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
            return image_data, box, clear_image_data
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        clearimg = clearimg.resize((nw, nh), Image.BICUBIC)
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image
        new_clearimg = Image.new('RGB', (w, h), (128, 128, 128))
        new_clearimg.paste(clearimg, (dx, dy))
        clearimg = new_clearimg
        flip = self.rand()<.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            clearimg = clearimg.transpose(Image.FLIP_LEFT_RIGHT)
        image_data      = np.array(image, np.uint8)
        clear_image_data = np.array(clearimg, np.uint8)
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        hue1, sat1, val1 = cv2.split(cv2.cvtColor(clear_image_data, cv2.COLOR_RGB2HSV))
        dtype1 = clear_image_data.dtype
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        x1 = np.arange(0, 256, dtype=r.dtype)
        lut_hue1 = ((x1 * r[0]) % 180).astype(dtype)
        lut_sat1 = np.clip(x1 * r[1], 0, 255).astype(dtype)
        lut_val1 = np.clip(x1 * r[2], 0, 255).astype(dtype)
        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        clear_image_data = cv2.merge((cv2.LUT(hue1, lut_hue1), cv2.LUT(sat1, lut_sat1), cv2.LUT(val1, lut_val1)))
        clear_image_data = cv2.cvtColor(clear_image_data, cv2.COLOR_HSV2RGB)
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)]
        return image_data, box, clear_image_data
    
    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx
                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx
                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox
        
def yolo_dataset_collate(batch):
    """
    Collate function supporting both single and dual fog views.
    """
    # Check if dual fog mode (4 items per batch element)
    if len(batch[0]) == 4:
        # Dual fog mode: (view1, view2, labels, clean)
        images_v1 = []
        images_v2 = []
        bboxes = []
        clearimg = []
        
        for i, (img_v1, img_v2, box, clear) in enumerate(batch):
            images_v1.append(img_v1)
            images_v2.append(img_v2)
            box[:, 0] = i
            bboxes.append(box)
            clearimg.append(clear)
        
        images_v1 = torch.from_numpy(np.array(images_v1)).type(torch.FloatTensor)
        images_v2 = torch.from_numpy(np.array(images_v2)).type(torch.FloatTensor)
        bboxes = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
        clearimg = torch.from_numpy(np.array(clearimg)).type(torch.FloatTensor)
        
        return images_v1, images_v2, bboxes, clearimg
    else:
        # Original single fog mode: (image, labels, clean)
        images  = []
        bboxes  = []
        clearimg = []
        for i, (img, box, clear) in enumerate(batch):
            images.append(img)
            box[:, 0] = i
            bboxes.append(box)
            clearimg.append(clear)
        images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
        bboxes  = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
        clearimg = torch.from_numpy(np.array(clearimg)).type(torch.FloatTensor)
        return images, bboxes, clearimg
