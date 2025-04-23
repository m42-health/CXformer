from torchxrayvision import datasets as xrv_datasets
import pandas as pd
import os
import numpy as np
from PIL import Image
import torch, torchvision
from cv2 import equalizeHist
import cv2
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import inspect, importlib
from typing import Optional, Callable, List, Union
import yaml
from torch.utils.data import ConcatDataset
import h5py, ast
from collections import Counter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json

SUPPORTED_DATASETS = [
    "CheX_Dataset",
    "CheX_Test_Dataset",
    "CheX_Binary_Test_Dataset",
    "NIH_Dataset",
    "Image Index",
    "RSNA_Pneumonia_Dataset",
    "Openi_Dataset",
    "Openi_Binary_Dataset",
    "SIIM_Pneumothorax_Dataset",
    "MIMIC_Dataset",
    "PC_Dataset",
    "VinDR_Dataset",
    "Montgomery",
    "Shenzhen",
    "ObjectCXR_Dataset",
    "BRAX",
    "Chest_DR",
    "CHSC",
    "INTBTR",
    "JSRT",
    "MIMIC_Segmentation_CheXmask",
    "SIIM_ACR_PNX_Segmentation",
    "VinDR_RibCXR_Segmentation"
]


def rle_decode(rle, shape=(1024, 1024)):
    """
    Decodes a run-length encoded array into a binary mask.

    Args:
        rle (numpy array): The run-length encoded data.
        shape (tuple): Shape of the mask (height, width).

    Returns:
        numpy array: Decoded binary mask.
    """
    # Initialize an empty mask
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    # RLE is a sequence of pairs (start, length)
    starts = rle[0::2] - 1  # Convert 1-based to 0-based indexing
    lengths = rle[1::2]

    # Iterate over each start and length to fill the mask
    for start, length in zip(starts, lengths):
        mask[start : start + length] = 1

    # Reshape to the original image shape
    return mask.reshape(shape)


def rle2mask_siim(rle, shape=(1024,1024)):
    width, height = shape
    mask= np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape(width, height)


def get_siim_mask(encoded_pixels, shape=(1024,1024)):
    if isinstance(encoded_pixels, list):
        mask = np.zeros(shape)
        for rle in encoded_pixels:
            if rle == '-1':
                continue
            mask += rle2mask_siim(rle, shape)
        return mask
    return rle2mask_siim(encoded_pixels, shape)

def get_mask(array, shape):
    mask = np.array(list(map(int, array.split())))
    mask = rle_decode(mask, shape=shape)
    return mask


class _NIH_Dataset(xrv_datasets.Dataset):
    def __init__(self,
                 imgpath,
                 csvpath=None,
                 bbox_list_path=None,
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 seed=0,
                 unique_patients=True,
                 pathology_masks=False,
                 pathologies=None
                 ):
        super(_NIH_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath

        self.csvpath = csvpath

        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks
        if pathologies:
            self.pathologies = pathologies
        else:
            self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                                "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                                "Effusion", "Pneumonia", "Pleural_Thickening",
                                "Cardiomegaly", "Nodule", "Mass", "Hernia"]

        self.pathologies = sorted(self.pathologies)

        # Load data
        self.csv = pd.read_csv(self.csvpath)

        # Remove images with view position other than specified
        self.csv["view"] = self.csv['View Position']
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("Patient ID").first()

        self.csv = self.csv.reset_index()

        ####### pathology masks ########
        # load nih pathology masks

        self.bbox_list_path = bbox_list_path
        if bbox_list_path:
            self.pathology_maskscsv = pd.read_csv(
                self.bbox_list_path,
                names=["Image Index", "Finding Label", "x", "y", "w", "h", "_1", "_2", "_3"],
                skiprows=1
            )

            # change label name to match
            self.pathology_maskscsv.loc[self.pathology_maskscsv["Finding Label"] == "Infiltrate", "Finding Label"] = "Infiltration"
            self.csv["has_masks"] = self.csv["Image Index"].isin(self.pathology_maskscsv["Image Index"])

        ####### pathology masks ########
        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            self.labels.append(self.csv["Finding Labels"].str.contains(pathology).values)

        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        # add consistent csv values

        # offset_day_int
        # self.csv["offset_day_int"] =

        # patientid
        self.csv["patientid"] = self.csv["Patient ID"].astype(str)

        # age
        self.csv['age_years'] = self.csv['Patient Age'] * 1.0

        # sex
        self.csv['sex_male'] = self.csv['Patient Gender'] == 'M'
        self.csv['sex_female'] = self.csv['Patient Gender'] == 'F'

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)


    def get_mask_dict(self, image_name, this_size):
        base_size = 1024
        scale = this_size / base_size

        images_with_masks = self.pathology_maskscsv[self.pathology_maskscsv["Image Index"] == image_name]
        path_mask = {}

        for i in range(len(images_with_masks)):
            row = images_with_masks.iloc[i]

            # Don't add masks for labels we don't have
            if row["Finding Label"] in self.pathologies:
                mask = np.zeros([this_size, this_size])
                xywh = np.asarray([row.x, row.y, row.w, row.h])
                xywh = xywh * scale
                xywh = xywh.astype(int)
                mask[xywh[1]:xywh[1] + xywh[3], xywh[0]:xywh[0] + xywh[2]] = 1

                # Resize so image resizing works
                mask = mask[None, :, :]

                path_mask[self.pathologies.index(row["Finding Label"])] = mask
        return path_mask

class MIMIC_Segmentation_CheXmask(Dataset):
    def __init__(
        self,
        imgpath,
        mimic_meta_csv,
        chexmask_csv,
        albumentations_transform=None,
        labels=None,
        img_size=518,
        seed=0,
        hist_equalize=False,
    ):
        df_mask = pd.read_csv(chexmask_csv)
        df_mimic = pd.read_csv(mimic_meta_csv)
        np.random.seed(seed)

        if isinstance(albumentations_transform, list):
            for i, t in enumerate(albumentations_transform):
                transform = [eval(t) if isinstance(t, str) else t for t in albumentations_transform]
            self.transform = A.Compose(transform)
        else:
            self.transform = albumentations_transform

        self.df = df_mask.merge(df_mimic, left_on="dicom_id", right_on="dicom_id")
        
        self.df["imgpath"] = self.df.apply(
            lambda row: os.path.join(
                imgpath,
                "p" + str(row["subject_id"])[:2],
                "p" + str(row["subject_id"]),
                "s" + str(row["study_id"]),
                row["dicom_id"] + ".jpg",
            ),
            axis=1,
        )
        self.hist_equalize = hist_equalize
        self.imgpath = imgpath

        if labels:
            self.labels = labels
            assert all(
                [label in self.df.columns for label in labels]
            ), f"Label(s) {labels} not found in the dataframe."
        else:
            self.labels = ["Left Lung", "Right Lung", "Heart"]
        self.df.dropna(subset=self.labels, inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        img = Image.open(self.df["imgpath"].loc[idx])
        img = np.array(img)

        if self.hist_equalize:
            img = equalizeHist(img)

        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for i, label in enumerate(self.labels, start=1):
            mask_label = get_mask(
                self.df[label].loc[idx], shape=(img.shape[0], img.shape[1])
            )
            mask[mask_label > 0] = (
                i  # Assign class index `i` where the mask is positive
            )
        

        # Convert to RGB, Albumentations won't work with grayscale images (https://github.com/albumentations-team/albumentations/issues/290#issuecomment-509293091)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.transform:
            data = self.transform(image=img, mask=mask)
            img = data["image"]
            mask = data["mask"]
        
        if isinstance(img, np.ndarray):
            if img.dtype == np.uint8:
                img = img / 255.0
        elif isinstance(img, torch.Tensor):
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
        
        # check if mask is numpy or tensor, and convert to int64
        if isinstance(mask, np.ndarray):
            mask = mask.astype(np.int64)
        elif isinstance(mask, torch.Tensor):
            mask = mask.to(torch.int64)

        return {
            "image": img,
            "mask": mask,
            "labels": {
                k: v for k, v in zip(self.labels, range(1, len(self.labels) + 1))
            },
        }

class VinDR_RibCXR_Segmentation(Dataset):
    def __init__(
        self,
        imgpath,
        json_annotation,
        albumentations_transform=None,
        labels=None,
        all_labels_as_one=True,
        seed=0,
        hist_equalize=False,
    ):
        with open(json_annotation) as f:
            data = json.load(f)
        
        self.df = pd.DataFrame(data)

        np.random.seed(seed)

        if isinstance(albumentations_transform, list):
            for i, t in enumerate(albumentations_transform):
                transform = [eval(t) if isinstance(t, str) else t for t in albumentations_transform]
            self.transform = A.Compose(transform)
        else:
            self.transform = albumentations_transform
        

        self.hist_equalize = hist_equalize
        self.imgpath = imgpath

        if labels:
            self.labels = labels
            assert all(
                [label in self.df.columns for label in labels]
            ), f"Label(s) {labels} not found in the dataframe."
        else:
            self.classes = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10']
            self.labels = self.classes
        
        self.df.dropna(subset=self.labels, inplace=True)
        self.df.reset_index(drop=True, inplace=True)


        if all_labels_as_one:
            self.labels = ['rib']

        self.all_labels_as_one = all_labels_as_one


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        img = Image.open(os.path.join(self.imgpath,self.df["img"].loc[idx]))
        img = img.convert("RGB")
        img = np.array(img)

        if self.hist_equalize:
            img = equalizeHist(img)

        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        

        if self.all_labels_as_one:
            for rib_name in self.classes:
                rib_points = self.df.iloc[idx][rib_name]  
                x_coords = [point['x'] for point in rib_points]
                y_coords = [point['y'] for point in rib_points]
                polygon_points = np.array(list(zip(x_coords, y_coords)), np.int32)  
                segmentation_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)   
                cv2.fillPoly(segmentation_mask, [polygon_points], 255)  # 255 is the color (white) for the mask   
                mask = cv2.add(mask, segmentation_mask)
            mask = np.where(mask > 0, 1, 0)

        else:
            masks = []
            for i,rib_name in enumerate(self.classes):
                rib_points = self.df.iloc[idx][rib_name]  
                x_coords = [point['x'] for point in rib_points]
                y_coords = [point['y'] for point in rib_points]
                polygon_points = np.array(list(zip(x_coords, y_coords)), np.int32)  
                mask_label = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)   
                cv2.fillPoly(mask_label, [polygon_points], 255)  # 255 is the color (white) for the mask   
                # masks.append(segmentation_mask)
                mask_label = np.where(mask_label > 0, 1, 0)
                masks.append(mask_label)    
            mask = np.stack(masks, axis=2)
            
        # for i, label in enumerate(self.labels):
        #     mask_label = get_mask(
        #         self.df[label].loc[idx], shape=(img.shape[0], img.shape[1])
        #     )
        #     mask[mask_label > 0] = (
        #         i  # Assign class index `i` where the mask is positive
        #     )
        

        # Convert to RGB, Albumentations won't work with grayscale images (https://github.com/albumentations-team/albumentations/issues/290#issuecomment-509293091)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.transform:
            # print(img.shape, mask.shape)
            data = self.transform(image=img, mask=mask)
            img = data["image"]
            mask = data["mask"]
        
        if isinstance(img, np.ndarray):
            if img.dtype == np.uint8:
                img = img / 255.0
        elif isinstance(img, torch.Tensor):
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
        
        # check if mask is numpy or tensor, and convert to int64
        if isinstance(mask, np.ndarray):
            mask = mask.astype(np.int64)
        elif isinstance(mask, torch.Tensor):
            mask = mask.to(torch.int64)

        return {
            "image": img,
            "mask": mask,
            "labels": {
                k: v for k, v in zip(self.labels, range(1, len(self.labels) + 1))
            },
        }




class SIIM_ACR_PNX_Segmentation(Dataset):
    def __init__(
        self,
        imgpath,
        dicomcsvpath,
        csvpath,
        albumentations_transform=None,
        labels=None,
        img_size=518,
        seed=0,
        hist_equalize=False,
    ):
        df_mask = pd.read_csv(csvpath)
        df_dicom = pd.read_csv(dicomcsvpath)
        np.random.seed(seed)

        if isinstance(albumentations_transform, list):
            for i, t in enumerate(albumentations_transform):
                transform = [eval(t) if isinstance(t, str) else t for t in albumentations_transform]
            self.transform = A.Compose(transform)
        else:
            self.transform = albumentations_transform

        self.df =  df_dicom.merge(df_mask, left_on='SOPInstanceUID', right_on='ImageId', how='right')

        def aggregate_encoded_pixels(series):
            if len(series) > 1:
                return list(series)
            return series.iloc[0]

        agg_dict = {col: 'first' for col in self.df.columns if col not in ['ImageId', ' EncodedPixels']}
        agg_dict[' EncodedPixels'] = aggregate_encoded_pixels
        self.df = self.df.groupby('ImageId').agg(agg_dict).reset_index()
        self.df["imgpath"] = self.df.apply(lambda x: os.path.join(imgpath, x['image_path'].replace('dicom-images-train/','').replace('.dcm','.jpg')), axis=1) 

        self.hist_equalize = hist_equalize
        self.imgpath = imgpath

        if labels:
            self.labels = labels
            assert all(
                [label in self.df.columns for label in labels]
            ), f"Label(s) {labels} not found in the dataframe."
        else:
            self.labels = ["Pneumothorax"]

        self.df.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        img = Image.open(self.df["imgpath"].loc[idx])
        img = np.array(img)

        if self.hist_equalize:
            img = equalizeHist(img)

        if self.df[" EncodedPixels"].loc[idx] == '-1':
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        else:
            mask = get_siim_mask(self.df[" EncodedPixels"].loc[idx], shape=(img.shape[0], img.shape[1]))


        # Convert to RGB, Albumentations won't work with grayscale images (https://github.com/albumentations-team/albumentations/issues/290#issuecomment-509293091)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.transform:
            data = self.transform(image=img, mask=mask)
            img = data["image"]
            mask = data["mask"]
        
        if isinstance(img, np.ndarray):
            if img.dtype == np.uint8:
                img = img / 255.0
        elif isinstance(img, torch.Tensor):
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
        
        # check if mask is numpy or tensor, and convert to int64
        if isinstance(mask, np.ndarray):
            mask = mask.astype(np.int64)
        elif isinstance(mask, torch.Tensor):
            mask = mask.to(torch.int64)

        return {
            "image": img,
            "mask": mask,
            "labels": {
                k: v for k, v in zip(self.labels, range(1, len(self.labels) + 1))
            },
        }


class _CheX_Dataset(xrv_datasets.Dataset):
    """CheXpert Dataset
    adapted from https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py
    """

    def __init__(
        self,
        imgpath,
        csvpath=None,
        views=["PA"],
        transform=None,
        data_aug=None,
        flat_dir=True,
        seed=0,
        unique_patients=False,
        pathologies=None,
    ):

        super(_CheX_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        if pathologies:
            self.pathologies = pathologies
        else:
            self.pathologies = [
                "Enlarged Cardiomediastinum",
                "Cardiomegaly",
                "Lung Opacity",
                "Lung Lesion",
                "Edema",
                "Consolidation",
                "Pneumonia",
                "Atelectasis",
                "Pneumothorax",
                "Pleural Effusion",
                "Pleural Other",
                "Fracture",
                "Support Devices",
            ]

        self.pathologies = sorted(self.pathologies)

        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.views = views

        self.csv["view"] = self.csv["Frontal/Lateral"]  # Assign view column
        self.csv.loc[(self.csv["view"] == "Frontal"), "view"] = self.csv[
            "AP/PA"
        ]  # If Frontal change with the corresponding value in the AP/PA column otherwise remains Lateral
        self.csv["view"] = self.csv["view"].replace(
            {"Lateral": "L"}
        )  # Rename Lateral with L

        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv["PatientID"] = self.csv["Path"].str.extract(pat=r"(patient\d+)")
            self.csv = self.csv.groupby("PatientID").first().reset_index()

        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                if pathology != "Support Devices":
                    self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]

            labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        # Make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = np.nan

        # Rename pathologies
        self.pathologies = list(
            np.char.replace(self.pathologies, "Pleural Effusion", "Effusion")
        )

        # patientid
        if "train" in self.csvpath:
            patientid = self.csv.Path.str.split("train/", expand=True)[1]
        elif "valid" in self.csvpath:
            patientid = self.csv.Path.str.split("valid/", expand=True)[1]
        else:
            raise NotImplementedError

        patientid = patientid.str.split("/study", expand=True)[0]
        patientid = patientid.str.replace("patient", "")

        # patientid
        self.csv["patientid"] = patientid

        # age
        self.csv["age_years"] = self.csv["Age"] * 1.0
        self.csv["Age"][(self.csv["Age"] == 0)] = None

        # sex
        self.csv["sex_male"] = self.csv["Sex"] == "Male"
        self.csv["sex_female"] = self.csv["Sex"] == "Female"

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(
            len(self), self.views, self.data_aug
        )

    def __len__(self):
        return len(self.labels)





class _INTBTR(Dataset):
    """
    Indian TB Dataset from https://intbcxr.nirt.res.in/

    """

    def __init__(self, imgpath, csvpath, seed=0):
        super(_INTBTR, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.csvpath = csvpath

        self.csv = pd.read_csv(csvpath)

        self.pathologies = [
            "tuberculosis",
        ]
        mask = self.csv["tuberculosis"]

        labels = []
        labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

    def string(self):
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.labels)


class _JSRT(Dataset):
    """
    Indian TB Dataset from https://intbcxr.nirt.res.in/

    """

    def __init__(self, imgpath, csvpath, seed=0):
        super(_JSRT, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.csvpath = csvpath

        self.csv = pd.read_csv(csvpath)

        self.pathologies = [
            "nodule",
        ]
        mask = self.csv["nodule"]

        labels = []
        labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

    def string(self):
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.labels)


class _CHSC(Dataset):
    """
    CHSC Private Data
    """

    def __init__(self, imgpath, csvpath, labels="binary", seed=0):
        super(_CHSC, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.csvpath = csvpath
        assert labels in {"binary", "multi"}, f"Unknown label type {labels}"

        self.csv = pd.read_csv(csvpath)

        if labels == "binary":
            self.pathologies = [
                "abnormal",
            ]
            mask = self.csv["BinaryEvaluation"]
        else:
            self.pathologies = ["normal", "abnormal_not_tb", "old_tb", "active_tb"]
            self.csv["MultiClassEvaluation"] = self.csv.apply(
                self.assign_multi_label_evaluation, axis=1
            )
            mask = self.csv["MultiClassEvaluation"]

        labels = []

        labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

    def string(self):
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.labels)

    def assign_multi_label_evaluation(self, row, strategy="majority"):
        label_map = {
            30: 0,
            31: 1,
            32: 2,
            33: 3,
        }
        assert strategy == "majority", f"Strategy {strategy} not implemented."
        if row.BinaryEvaluation == 0:
            return 0
        votes = row.FinalEvaluation.split(",")
        votes = [int(vote) for vote in votes]
        votes = [33 if vote in [33, 34, 35, 36, 37] else vote for vote in votes]
        counts = Counter(votes)
        max_count = max(counts.values())
        candidates = [item for item, count in counts.items() if count == max_count]
        final_vote = max(candidates)
        return label_map[final_vote]


class _Chest_DR(Dataset):
    """Chest_DR Dataset
    Adapted from https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py
    """

    def __init__(self, imgpath, csvpath, seed=0):
        super(_Chest_DR, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.csvpath = csvpath
        self.views = []
        self.pathologies = [
            "pleural_effusion",
            "nodule",
            "pneumonia",
            "cardiomegaly",
            "hilar_enlargement",
            "fracture_old",
            "fibrosis",
            "aortic_calcification",
            "tortuous_aorta",
            "thickened_pleura",
            "TB",
            "pneumothorax",
            "emphysema",
            "atelectasis",
            "calcification",
            "pulmonary_edema",
            "increased_lung_markings",
            "elevated_diaphragm",
            "consolidation",
        ]
        self.pathologies = sorted(self.pathologies)
        self.csv = pd.read_csv(csvpath)

        labels = []

        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                mask = self.csv[pathology]
            labels.append(mask.values)

        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={}".format(
            len(self), self.views
        )

    def __len__(self):
        return len(self.labels)


class _ObjectCXR_Dataset(Dataset):
    """ObjectCXR Dataset
    Adapted from https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py
    """

    def __init__(self, imgpath, csvpath, seed=0):
        super(_ObjectCXR_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.csvpath = csvpath
        self.views = []
        self.pathologies = ["Foreign Object"]

        # Load data
        self.csv = pd.read_csv(self.csvpath)

        labels = []
        labels.append(~self.csv["annotation"].isnull())
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        self.csv = self.csv.reset_index()

        self.csv["has_masks"] = ~self.csv["annotation"].isnull()

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={}".format(
            len(self), self.views
        )

    def __len__(self):
        return len(self.labels)


class _NLMTB_Dataset(xrv_datasets.Dataset):
    """National Library of Medicine Tuberculosis Datasets
    Adapted from https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py
    """

    def __init__(self, imgpath, seed=0, views=["PA"]):
        """
        Args:
            img_path (str): Path to `MontgomerySet` or `ChinaSet_AllFiles`
                folder
        """
        super(_NLMTB_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath

        file_list = []

        for fname in sorted(os.listdir(os.path.join(self.imgpath))):
            if fname.endswith(".png") or fname.endswith(".jpg"):
                file_list.append(fname)

        self.csv = pd.DataFrame({"fname": file_list})

        # Label is the last digit on the simage filename
        self.csv["label"] = self.csv["fname"].apply(lambda x: int(x.split(".")[-2][-1]))
        # All the images are PA according to the article.
        self.csv["view"] = "PA"
        self.limit_to_selected_views(views)

        self.labels = self.csv["label"].values.reshape(-1, 1)
        self.pathologies = ["Tuberculosis"]

    def __len__(self):
        return len(self.csv)

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={}".format(
            len(self), self.views
        )


class _BRAX(xrv_datasets.Dataset):
    """BRAX Dataset
    Adapted from https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py
    """

    def __init__(self, imgpath, csvpath, seed=0, views=["PA"], unique_patients=False):
        """
        Args:
            img_path (str): Path to BRAX dataset image folder
        """
        super(_BRAX, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.view = views
        self.pathologies = [
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Lesion",
            "Lung Opacity",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
        ]
        self.pathologies = sorted(self.pathologies)
        self.csv = pd.read_csv(csvpath)
        self.csv.rename(columns={"ViewPosition": "view"}, inplace=True)
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("PatientID").first().reset_index()

        healthy = self.csv["No Finding"] == 1
        labels = []

        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                if pathology != "Support Devices":
                    self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]

            labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        # Make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = np.nan
        self.csv["imgid"] = self.csv["PngPath"].apply(
            lambda x: x.split("images/")[-1].replace(".png", "")
        )

    def __len__(self):
        return len(self.csv)

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={}".format(
            len(self), self.views
        )


class _CheX_Binary_Test_Dataset(xrv_datasets.Dataset):
    """CheXpert Test Dataset
    Adapted from https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py
    """

    def __init__(
        self,
        imgpath,
        csvpath,
        seed=0,
        views=["PA"],
        unique_patients=False,
        pathologies=None,
    ):
        """
        Args:
            img_path (str): Path to Chexpert dataset image folder
        """
        super(_CheX_Binary_Test_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.view = views
        if pathologies:
            self.pathologies = pathologies
        else:
            self.pathologies = ["abnormal"]

        self.csv = pd.read_csv(csvpath)
        self.csv["view"] = self.csv.Path.apply(
            lambda x: x.split("/")[-1].replace(".jpg", "").split("_")[-1].capitalize()
        )
        for v in views:
            assert v in ("Frontal", "Lateral")
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv["PatientID"] = self.csv["Path"].str.extract(pat=r"(patient\d+)")
            self.csv = self.csv.groupby("PatientID").first().reset_index()

        mask = self.csv["abnormal"]
        labels = []
        labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

    def __len__(self):
        return len(self.csv)

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={}".format(
            len(self), self.views
        )


class _CheX_Test_Dataset(xrv_datasets.Dataset):
    """CheXpert Test Dataset
    Adapted from https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py
    """

    def __init__(
        self,
        imgpath,
        csvpath,
        seed=0,
        views=["PA"],
        unique_patients=False,
        pathologies=None,
    ):
        """
        Args:
            img_path (str): Path to BRAX dataset image folder
        """
        super(_CheX_Test_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.view = views
        if pathologies:
            self.pathologies = pathologies
        else:
            self.pathologies = [
                "Enlarged Cardiomediastinum",
                "Cardiomegaly",
                "Lung Opacity",
                "Lung Lesion",
                "Edema",
                "Consolidation",
                "Pneumonia",
                "Atelectasis",
                "Pneumothorax",
                "Pleural Effusion",
                "Pleural Other",
                "Fracture",
                "Support Devices",
            ]

        self.pathologies = sorted(self.pathologies)
        self.csv = pd.read_csv(csvpath)
        self.csv["view"] = self.csv.Path.apply(
            lambda x: x.split("/")[-1].replace(".jpg", "").split("_")[-1].capitalize()
        )
        for v in views:
            assert v in ("Frontal", "Lateral")
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv["PatientID"] = self.csv["Path"].str.extract(pat=r"(patient\d+)")
            self.csv = self.csv.groupby("PatientID").first().reset_index()

        healthy = self.csv["No Finding"] == 1
        labels = []

        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                if pathology != "Support Devices":
                    self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]

            labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        # Rename pathologies
        self.pathologies = list(
            np.char.replace(self.pathologies, "Pleural Effusion", "Effusion")
        )

        # Make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = np.nan
        # self.csv['imgid'] = self.csv['Path']

    def __len__(self):
        return len(self.csv)

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={}".format(
            len(self), self.views
        )


class _VinDR_Dataset(xrv_datasets.Dataset):
    """CheXpert Test Dataset
    Adapted from https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py
    """

    def __init__(
        self,
        imgpath,
        csvpath,
        seed=0,
        pathologies=None,
        label_aggregate_strategy="majority",
    ):
        """
        Args:
            img_path (str): Path to BRAX dataset image folder
        """
        super(_VinDR_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        if not pathologies:
            self.pathologies = [
                "Aortic enlargement",
                "Atelectasis",
                "Calcification",
                "Cardiomegaly",
                "Clavicle fracture",
                "Consolidation",
                "Edema",
                "Emphysema",
                "Enlarged PA",
                "ILD",
                "Infiltration",
                "Lung Opacity",
                "Lung cavity",
                "Lung cyst",
                "Mediastinal shift",
                "Nodule/Mass",
                "Pleural effusion",
                "Pleural thickening",
                "Pneumothorax",
                "Pulmonary fibrosis",
                "Rib fracture",
                "Other lesion",
                "COPD",
                "Lung tumor",
                "Pneumonia",
                "Tuberculosis",
                "Other diseases",
                "No finding",
            ]
        else:
            self.pathologies = pathologies
        self.pathologies = sorted(self.pathologies)
        self.csv = pd.read_csv(csvpath)

        def majority_vote(series):
            return series.mode().iloc[0]

        def atleast_one(series):
            return 1 if any(series == 1) else 0

        def join_strings(series):
            return list(series)

        if label_aggregate_strategy == "majority":
            aggregation_strategy = majority_vote
        elif label_aggregate_strategy == "atleast_one":
            aggregation_strategy = atleast_one
        else:
            raise ValueError(
                f"Unknown label aggregation strategy {label_aggregate_strategy}"
            )

        columns_aggregation = {
            pathology: aggregation_strategy for pathology in self.pathologies
        }

        self.csv = self.csv.groupby("image_id").agg(columns_aggregation).reset_index()

        labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                mask = self.csv[pathology]
            labels.append(mask.values)

        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

    def __len__(self):
        return len(self.csv)

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={}".format(
            len(self), self.views
        )


class _Openi_Binary_Dataset(xrv_datasets.Dataset):
    def __init__(self, imgpath, csvpath, seed=0):
        super(_Openi_Binary_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.csvpath = csvpath

        self.csv = pd.read_csv(csvpath)

        self.pathologies = [
            "abnormal",
        ]
        mask = self.csv["abnormal"]

        labels = []

        labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

    def string(self):
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.labels)


class _RSNA_Pneumonia_Dataset(xrv_datasets.Dataset):
    """RSNA Pneumonia Dataset
    Adapted from https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py
    """

    def __init__(
        self,
        imgpath,
        csvpath,
        dicomcsvpath,
        views=["PA"],
        transform=None,
        data_aug=None,
        nrows=None,
        seed=0,
    ):
        super(_RSNA_Pneumonia_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.pathologies = ["Pneumonia"]

        self.raw_csv = pd.read_csv(csvpath, nrows=nrows)
        self.dicomcsv = pd.read_csv(dicomcsvpath, nrows=nrows, index_col="PatientID")
        self.csv = pd.merge(
            self.raw_csv, self.dicomcsv, left_on="patientId", right_on="PatientID"
        )

        self.csv["view"] = self.csv["ViewPosition"]
        self.limit_to_selected_views(views)

        self.csv = (
            self.csv.groupby("patientId")
            .agg(
                {
                    "x": list,
                    "y": list,
                    "width": list,
                    "height": list,
                    "ViewPosition": "first",
                    "PatientAge": "first",
                    "PatientSex": "first",
                    "SOPInstanceUID": "first",
                    "StudyInstanceUID": "first",
                    "Target": list,
                }
            )
            .reset_index()
        )
        self.csv["Target"] = self.csv["Target"].apply(
            lambda x: max(set(x), key=x.count)
        )

        labels = [self.csv["Target"].values]
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

    def __len__(self):
        return len(self.csv)

    def __str__(self):
        return "RSNA Pneumonia Dataset, len={}".format(len(self))


def get_dataclass(dataset_name: str):
    """
    Returns the XRV data class for the specified dataset name.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        data_class: The TorchXrayVision data class for the specified dataset.

    """
    if dataset_name in {"Shenzhen", "Montgomery"}:
        return _NLMTB_Dataset
    elif dataset_name == "CHSC":
        return _CHSC
    elif dataset_name == "ObjectCXR_Dataset":
        return _ObjectCXR_Dataset
    elif dataset_name == "BRAX":
        return _BRAX
    elif dataset_name == "CheX_Dataset":
        return _CheX_Dataset
    elif dataset_name == "CheX_Test_Dataset":
        return _CheX_Test_Dataset
    elif dataset_name == "CheX_Binary_Test_Dataset":
        return _CheX_Binary_Test_Dataset
    elif dataset_name == "Chest_DR":
        return _Chest_DR
    elif dataset_name == "VinDR_Dataset":
        return _VinDR_Dataset
    elif dataset_name == "RSNA_Pneumonia_Dataset":
        return _RSNA_Pneumonia_Dataset
    elif dataset_name == "Openi_Binary_Dataset":
        return _Openi_Binary_Dataset
    elif dataset_name == "INTBTR":
        return _INTBTR
    elif dataset_name == "JSRT":
        return _JSRT
    elif dataset_name == "MIMIC_Segmentation_CheXmask":
        return MIMIC_Segmentation_CheXmask
    elif dataset_name == "SIIM_ACR_PNX_Segmentation":
        return SIIM_ACR_PNX_Segmentation
    elif dataset_name == "NIH_Dataset":
        return _NIH_Dataset
    elif dataset_name == 'VinDR_RibCXR_Segmentation':
        return VinDR_RibCXR_Segmentation
    module = importlib.import_module("torchxrayvision.datasets")
    return getattr(module, dataset_name)


def compute_cdf(histogram):
    cdf = histogram.cumsum()
    cdf_normalized = cdf / cdf[-1]  # Normalize to [0, 1]
    return cdf_normalized


def histogram_matching(target_image, reference_cdf):
    target_hist, _ = np.histogram(target_image.flatten(), bins=256, range=[0, 256])
    target_cdf = compute_cdf(target_hist)
    lookup_table = np.interp(target_cdf, reference_cdf, np.arange(256))
    matched_image = lookup_table[target_image.flatten()].reshape(target_image.shape)
    return np.uint8(matched_image)


def read_xray_image(
    jpg_path: str,
    hist_equalize: bool = True,
    transform: Optional[Callable] = None,
    reference_cdf=None,
) -> torch.FloatTensor:
    """
    Read an X-ray image from the given path and perform optional histogram equalization and transformation.

    Args:
        jpg_path (str): The path to the JPEG image file.
        hist_equalize (bool): Whether to perform histogram equalization on the image (default: True).
        transform (Optional[Callable]): Optional transformation function to apply to the image (default: None).

    Returns:
        torch.FloatTensor: The processed image as a tensor.

    """
    image = Image.open(jpg_path)
    image = image.convert("L")  # convert to grayscale
    if reference_cdf is not None:
        image = histogram_matching(np.array(image), reference_cdf)
        image = Image.fromarray(image)

    if hist_equalize:
        image = np.array(image)
        image = equalizeHist(image)
        image = Image.fromarray(image)

    if transform:
        image = transform(image)

    # if not isinstance(image, torch.Tensor):
    #     image = F.to_tensor(image)
    return image


def read_xray_image_from_h5(
    img_dset,
    img_paths_dset,
    jpg_path: str,
    hist_equalize: bool = True,
    transform: Optional[Callable] = None,
) -> torch.FloatTensor:
    """
    Read an X-ray image from the given path and perform optional histogram equalization and transformation.

    Args:
        h5_ds : H5 Dataset object
        jpg_path (str): Relative path to the JPEG image file inside H5 file.
        hist_equalize (bool): Whether to perform histogram equalization on the image (default: True).
        transform (Optional[Callable]): Optional transformation function to apply to the image (default: None).

    Returns:
        torch.FloatTensor: The processed image as a tensor.

    """
    path_to_idx_str = img_paths_dset.attrs.get("path_to_idx", "{}")
    path_to_idx = ast.literal_eval(path_to_idx_str)

    # Find index of the image path
    idx = path_to_idx.get(jpg_path, None)

    if idx is None:
        raise ValueError(f"Image path '{jpg_path}' not found in HDF5 file.")

    # Retrieve the image from the dataset
    image = img_dset[idx]

    # convert image to PIL image
    image = Image.fromarray(image)
    image = image.convert("L")  # convert to grayscale

    if hist_equalize:
        image = np.array(image)
        image = equalizeHist(image)
        image = Image.fromarray(image)

    if transform:
        image = transform(image)

    return image


class CXRDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        imgpath: str,
        transform: Optional[Union[Callable, List]] = None,
        data_aug: Optional[Callable] = None,
        seed: int = 42,
        hist_equalize: bool = True,
        uncertain_labels_method: str = "U-ONES",
        reference_histogram: Optional[str] = None,
        skip_image: bool = False,
        **kwargs,
    ):
        """
        Initializes a CXR Dataset object, using TorchXrayVision implementation.

        Args:
            dataset_name (str): The name of the dataset.
            imgpath (str): The path to the directory containing the images.
            csvpath (str, optional): The path to the CSV file containing metadata. Defaults to None.
            views (List[str], optional): The list of views to include. Defaults to ["PA"].
            transform (Callable or list, optional): A function to apply transformations to the images. Defaults to None.
            data_aug (Callable, optional): A function to apply data augmentation to the images. Defaults to None.
            seed (int, optional): The random seed. Defaults to 0.
            unique_patients (bool, optional): Whether to use unique patients only. Defaults to False.
            hist_equalize (bool, optional): Whether to perform histogram equalization. Defaults to True.
        """
        assert uncertain_labels_method in {
            "U-ONES",
            "U-ZEROS",
        }, f"Uknown {uncertain_labels_method} selected for resolving uncertain labels."
        if dataset_name not in SUPPORTED_DATASETS:
            raise NotImplementedError(f"{dataset_name} is not supported.")

        if imgpath.endswith(".h5"):
            dset = h5py.File(imgpath, "r")
            self.img_dset = dset["cxr"]
            self.img_path_mapper_dset = dset["cxr_paths"]

        self.dataset = get_dataclass(dataset_name)(
            imgpath=imgpath,
            seed=seed,
            **kwargs,
        )
        self.dataset_name = dataset_name
        self.uncertain_labels_method = uncertain_labels_method
        self.imgpath = imgpath
        self.skip_image = skip_image
        if transform:
            assert (
                not self.dataset_name in ["MIMIC_Segmentation_CheXmask", "SIIM_ACR_PNX_Segmentation"]
            ), "MIMIC_Segmentation_CheXmask does not support torchvision transforms use 'albumentations_transforms' key instead"
        if isinstance(transform, list):
            for i, t in enumerate(transform):
                transform = [eval(t) if isinstance(t, str) else t for t in transform]
            transform = torchvision.transforms.Compose(transform)
        self.transform = transform
        self.data_aug = data_aug
        self.hist_equalize = hist_equalize
        self.reference_histogram = (
            np.load(reference_histogram) if reference_histogram else None
        )

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, index):
        sample = {}
        image_path = None
        misc = "None"

        if self.dataset_name == "CheX_Dataset":  # chexpert
            imgid = self.dataset.csv["Path"].iloc[index]
        elif self.dataset_name == "CheX_Test_Dataset":  # chexpert test
            imgid = self.dataset.csv["Path"].iloc[index]
        elif self.dataset_name == "CheX_Binary_Test_Dataset":  # chexpert test
            imgid = self.dataset.csv["Path"].iloc[index]
        elif self.dataset_name == "NIH_Dataset":  # cxr14
            imgid = self.dataset.csv["Image Index"].iloc[index]
        elif self.dataset_name == "RSNA_Pneumonia_Dataset":
            imgid = self.dataset.csv["patientId"].iloc[index]
        elif self.dataset_name == "Openi_Dataset":
            imgid = self.dataset.csv["imageid"].iloc[index]
        elif self.dataset_name == "Openi_Binary_Dataset":
            imgid = self.dataset.csv["path"].iloc[index]
        elif self.dataset_name == "SIIM_Pneumothorax_Dataset":
            imgid = self.dataset.csv["ImageId"].iloc[index]
            image_path = self.dataset.file_map[imgid + ".jpg"]
        elif self.dataset_name == "MIMIC_Dataset":
            subjectid = str(self.dataset.csv.iloc[index]["subject_id"])
            studyid = str(self.dataset.csv.iloc[index]["study_id"])
            dicom_id = str(self.dataset.csv.iloc[index]["dicom_id"])
            image_path = os.path.join(
                self.imgpath,
                "p" + subjectid[:2],
                "p" + subjectid,
                "s" + studyid,
                dicom_id + ".jpg",
            )
        elif self.dataset_name == "PC_Dataset":
            imgid = self.dataset.csv["ImageID"].iloc[index]
            image_dir = str(self.dataset.csv["ImageDir"].iloc[index])
            image_path = os.path.join(self.imgpath, image_dir, imgid)
        elif self.dataset_name == "VinDR_Dataset":  # vindr
            imgid = self.dataset.csv["image_id"].iloc[index]
        elif self.dataset_name == "Montgomery" or self.dataset_name == "Shenzhen":
            imgid = self.dataset.csv.iloc[index]["fname"]
        elif self.dataset_name == "ObjectCXR_Dataset":
            imgid = self.dataset.csv.iloc[index]["image_name"]
        elif self.dataset_name == "BRAX":
            imgid = self.dataset.csv.iloc[index]["imgid"]
        elif self.dataset_name == "Chest_DR":
            imgid = self.dataset.csv.iloc[index]["img_id"]
        elif self.dataset_name == "CHSC":
            imgid = self.dataset.csv.iloc[index]["jpg_fpath"]
            try:
                misc = self.dataset.csv.iloc[index]["WHO region"]
            except:
                pass
        elif self.dataset_name == "INTBTR":
            imgid = self.dataset.csv.iloc[index]["jpg_fpath"]
        elif self.dataset_name == "JSRT":
            imgid = self.dataset.csv.iloc[index]["image"]
        elif self.dataset_name == "MIMIC_Segmentation_CheXmask":
            return self.dataset[index]
        elif self.dataset_name == "SIIM_ACR_PNX_Segmentation":
            return self.dataset[index]
        elif self.dataset_name == "VinDR_RibCXR_Segmentation":
            return self.dataset[index]
        else:
            raise NotImplementedError(f"{self.dataset_name} not implemented.")
        if not image_path:
            image_path = os.path.join(self.imgpath, imgid)

        image_path = os.path.splitext(image_path)[0] + ".jpg"

        if self.imgpath.endswith(".h5"):
            rel_path = os.path.splitext(imgid)[0] + ".jpg"
            image = read_xray_image_from_h5(
                self.img_dset,
                self.img_path_mapper_dset,
                rel_path,
                hist_equalize=self.hist_equalize,
                transform=self.transform,
            )
            sample["path"] = rel_path
        else:
            if self.skip_image:
                image = "None"
            else:
                image = read_xray_image(
                    image_path,
                    hist_equalize=self.hist_equalize,
                    transform=self.transform,
                    reference_cdf=self.reference_histogram,
                )
            sample["path"] = image_path

        sample["idx"] = index
        sample["image"] = image
        sample["lab"] = self.dataset.labels[index]

        if (
            self.dataset_name == "CheX_Dataset"
            and self.uncertain_labels_method == "U-ONES"
        ):  # Resolve uncertain labels for CheXpert
            sample["lab"] = np.nan_to_num(sample["lab"], nan=1)
        elif (
            self.dataset_name == "CheX_Dataset"
            and self.uncertain_labels_method == "U-ZEROS"
        ):
            sample["lab"] = np.nan_to_num(sample["lab"], nan=0)

        sample["dataset"] = self.dataset_name
        sample["misc"] = misc
        return sample

    def __len__(self):
        return len(self.dataset)

    def __str__(self):
        return str(self.dataset)


def get_dataset_from_yaml(yaml_path: str):
    """
    Returns a CXRDataset object from the specified YAML file.

    Args:
        yaml_path (str): The path to the YAML file.

    Returns:
        CXRDataset: The CXRDataset object.

    """
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    if "datasets" in config:
        config = config["datasets"]
    if len(config) == 1:
        config = config[0]
        return CXRDataset(**config)
    dataset_list = [CXRDataset(**dataset_args) for dataset_args in config]
    return ConcatDataset(dataset_list)
