#!/usr/env/bin python3.6

import io
import re
import random
from operator import itemgetter
from pathlib import Path
from itertools import repeat
from functools import partial
from typing import Any, Callable, BinaryIO, Dict, List, Match, Pattern, Tuple, Union

import torch
import numpy as np
from torch import Tensor
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler

from utils import id_, map_, class2one_hot
from utils import simplex, sset, one_hot

F = Union[Path, BinaryIO]
D = Union[Image.Image, np.ndarray, Tensor]


def get_loaders(args, data_folder: str, weak_subfolder: str, dict_params: Dict,
                batch_size: int, n_class: int,
                debug: bool, in_memory: bool, dtype) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: nd / 255,  # max <= 1
        lambda nd: torch.tensor(nd, dtype=dtype)
    ])
    gt_transform = transforms.Compose([
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: torch.tensor(nd, dtype=torch.int64),
        partial(class2one_hot, C=n_class),
        itemgetter(0)
    ])

    losses = eval(args.losses)
    bounds_generators: List[Callable] = []
    for _, _, bounds_name, bounds_params, fn, _ in losses:
        if bounds_name is None:
            bounds_generators.append(lambda *a: torch.zeros(n_class, 1, 2))
            continue

        bounds_class = getattr(__import__('bounds'), bounds_name)
        bounds_generators.append(bounds_class(C=args.n_class, fn=fn, **bounds_params))

    # Create partial functions: Easier for readability later (see the difference between train and validation)
    gen_dataset = partial(SliceDataset,
                          transforms=[transform, gt_transform, gt_transform],
                          debug=debug,
                          C=n_class,
                          dict_params=dict_params,
                          in_memory=in_memory,
                          bounds_generators=bounds_generators)
    data_loader = partial(DataLoader,
                          num_workers=10,
                          pin_memory=True)

    # Prepare the datasets and dataloaders
    train_folders: List[Path] = [Path(data_folder, "train", "img"),
                                 Path(data_folder, "train", "gt"),
                                 Path(data_folder, "train", weak_subfolder)]
    # I assume all files have the same name inside their folder: makes things much easier
    train_names: List[str] = map_(lambda p: str(p.name), train_folders[0].glob("*.png"))
    train_set = gen_dataset(train_names,
                            train_folders)
    train_loader = data_loader(train_set,
                               batch_size=batch_size,
                               shuffle=True,
                               drop_last=True)

    val_folders: List[Path] = [Path(data_folder, "val", "img"),
                               Path(data_folder, "val", "gt"),
                               Path(data_folder, "val", weak_subfolder)]
    val_names: List[str] = map_(lambda p: str(p.name), val_folders[0].glob("*.png"))
    val_set = gen_dataset(val_names,
                          val_folders)
    val_sampler = PatientSampler(val_set, args.grp_regex, shuffle=False)
    # val_sampler = None
    val_loader = data_loader(val_set,
                             batch_sampler=val_sampler)

    return train_loader, val_loader


class SliceDataset(Dataset):
    def __init__(self, filenames: List[str], folders: List[Path], bounds_generators: List[Callable],
                 transforms=None, equalize=False, debug=False,
                 C=4, dict_params=None, in_memory: bool = False) -> None:
        self.folders: List[Path] = folders
        self.transforms: List[Callable[[D], Tensor]] = transforms if transforms else [id_] * len(folders)
        assert len(self.transforms) == len(self.folders)

        self.filenames: List[str] = filenames
        self.equalize = equalize
        self.debug = debug
        self.C: int = C  # Number of classes
        self.in_memory: bool = in_memory
        self.bounds_generators: List[Callable] = bounds_generators

        if self.debug:
            self.filenames = self.filenames[:10]

        assert self.check_files()  # Make sure all file exists

        # Load things in memory if needed
        self.files: List[List[F]] = SliceDataset.load_images(self.folders, self.filenames, self.in_memory)
        assert len(self.files) == len(self.folders)
        for files in self.files:
            assert len(files) == len(self.filenames)

        print(f"Initialized {self.__class__.__name__} with {len(self.filenames)} images and params {dict_params}")

    def check_files(self) -> bool:
        for folder in self.folders:
            for f_n in self.filenames:
                if not Path(folder, f_n).exists():
                    return False

        return True

    @staticmethod
    def load_images(folders: List[Path], filenames: List[str], in_memory: bool) -> List[List[F]]:
        def load(folder: Path, filename: str) -> F:
            p: Path = Path(folder, filename)
            if in_memory:
                with open(p, 'rb') as data:
                    res = io.BytesIO(data.read())
                return res
            return p
        if in_memory:
            print("Loading the data in memory...")

        files: List[List[F]] = [[load(f, im) for im in filenames] for f in folders]

        return files

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int) -> List[Any]:
        filename: str = self.filenames[index]
        path_name: Path = Path(filename)
        images: List[D]
        if path_name.suffix == ".png":
            images = [Image.open(files[index]).convert('L') for files in self.files]
        elif path_name.suffix == ".npy":
            images = [np.load(files[index]) for files in self.files]
        else:
            raise ValueError(filename)

        # Final transforms and assertions
        t_tensors: List[Tensor] = [tr(e) for (tr, e) in zip(self.transforms, images)]

        assert 0 <= t_tensors[0].min() and t_tensors[0].max() <= 1  # main image is between 0 and 1
        b, w, h = t_tensors[0].shape
        assert b == 1
        for ttensor in t_tensors[1:]:  # All masks (ground truths) are class encoded
            assert one_hot(ttensor, axis=0)
            assert ttensor.shape == (self.C, w, h)

        bounds = [f(*t_tensors, filename) for f in self.bounds_generators]

        # return t_tensors + [filename] + bounds
        return [filename] + t_tensors + bounds


class PatientSampler(Sampler):
    def __init__(self, dataset: SliceDataset, grp_regex, shuffle=False) -> None:
        filenames: List[str] = dataset.filenames
        # Might be needed in case of escape sequence fuckups
        # self.grp_regex = bytes(grp_regex, "utf-8").decode('unicode_escape')
        self.grp_regex = grp_regex

        # Configure the shuffling function
        self.shuffle: bool = shuffle
        self.shuffle_fn: Callable = (lambda x: random.sample(x, len(x))) if self.shuffle else id_

        print(f"Grouping using {self.grp_regex} regex")
        # assert grp_regex == "(patient\d+_\d+)_\d+"
        # grouping_regex: Pattern = re.compile("grp_regex")
        grouping_regex: Pattern = re.compile(self.grp_regex)

        stems: List[str] = [Path(filename).stem for filename in filenames]  # avoid matching the extension
        matches: List[Match] = map_(grouping_regex.match, stems)
        patients: List[str] = [match.group(1) for match in matches]

        unique_patients: List[str] = list(set(patients))
        assert len(unique_patients) < len(filenames)
        print(f"Found {len(unique_patients)} unique patients out of {len(filenames)} images")

        self.idx_map: Dict[str, List[int]] = dict(zip(unique_patients, repeat(None)))
        for i, patient in enumerate(patients):
            if not self.idx_map[patient]:
                self.idx_map[patient] = []

            self.idx_map[patient] += [i]
        # print(self.idx_map)
        assert sum(len(self.idx_map[k]) for k in unique_patients) == len(filenames)

        print("Patient to slices mapping done")

    def __len__(self):
        return len(self.idx_map.keys())

    def __iter__(self):
        values = list(self.idx_map.values())
        shuffled = self.shuffle_fn(values)
        return iter(shuffled)
