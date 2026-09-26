"""TinyImageNet-200, and its fine-to-coarse (WordNet superclass) label mapping.

The labelled validation split is the test set (TinyImageNet's own test split is
unlabelled). It is read either as shipped -- flat ``val/images`` with labels in
``val_annotations.txt`` -- or already sorted into ``val/<wnid>/`` folders; files
are never moved.

TinyImageNet's classes are WordNet noun synsets, so unlike CIFAR-10 the coarse
grouping does not have to be invented: it is read off the hypernym graph. Each
class starts as its own group and is lifted to its first hypernym until every
group holds at least two classes, which is the property subclass deletion needs
-- the superclass has to survive the deletion through a sibling. The result is
56 superclasses of 2-12 classes each, no singletons.

The mapping is computed once with NLTK's WordNet and cached as JSON next to the
dataset, so runs do not depend on NLTK being installed and every seed and cell
sees byte-identical superclasses. Class ids follow ``ImageFolder`` order, i.e.
the sorted wnid directory names, not the order in ``wnids.txt``.
"""

import collections
import copy
import json
import os

from torchvision import datasets, transforms

CACHE_NAME = 'tinyimagenet-200-coarse.json'


class FlatValidation(datasets.VisionDataset):
    """The shipped flat validation split, in the order ``ImageFolder`` would give."""

    def __init__(self, root, classes, transform=None):
        super().__init__(root, transform=transform)
        index = {wnid: i for i, wnid in enumerate(classes)}
        with open(os.path.join(root, 'val_annotations.txt')) as handle:
            rows = sorted((index[line.split('\t')[1]], line.split('\t')[0]) for line in handle)
        self.samples = [(os.path.join(root, 'images', name), target) for target, name in rows]
        self.targets = [target for target, _ in rows]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = datasets.folder.default_loader(path)
        return (self.transform(image) if self.transform else image), target


class TinyImageNetDataset:
    num_classes = 200
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    def __init__(self, root='./data', val_split=0.1, seed=42):
        root = os.path.join(root, 'tiny-imagenet-200')
        if not os.path.isdir(os.path.join(root, 'train')):
            raise FileNotFoundError(
                f'TinyImageNet not found at {root}; download '
                'http://cs231n.stanford.edu/tiny-imagenet-200.zip and extract it there'
            )
        self.root = root
        self.val_split = val_split
        normalize = transforms.Normalize(self.mean, self.std)
        self.train_transform = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        train = os.path.join(root, 'train')
        self.train_dataset = datasets.ImageFolder(train, transform=self.train_transform)
        self.val_dataset = datasets.ImageFolder(train, transform=self.test_transform)
        val = os.path.join(root, 'val')
        self.test_dataset = (
            FlatValidation(val, self.train_dataset.classes, self.test_transform)
            if os.path.isdir(os.path.join(val, 'images'))
            else datasets.ImageFolder(val, transform=self.test_transform)
        )


def _build(root):
    from nltk.corpus import wordnet

    wnids = sorted(os.listdir(os.path.join(root, 'train')))
    synsets = {wnid: wordnet.synset_from_pos_and_offset('n', int(wnid[1:])) for wnid in wnids}
    key = dict(synsets)
    for _ in range(30):
        groups = collections.defaultdict(list)
        for wnid, synset in key.items():
            groups[synset].append(wnid)
        small = [synset for synset, members in groups.items() if len(members) < 2]
        if not small:
            break
        for synset in small:
            hypernyms = synset.hypernyms() or synset.instance_hypernyms()
            if hypernyms:
                for wnid in groups[synset]:
                    key[wnid] = hypernyms[0]
    coarse_names = sorted({synset.name() for synset in key.values()})
    coarse_index = {name: index for index, name in enumerate(coarse_names)}
    return {
        'wnids': wnids,
        'coarse_names': coarse_names,
        'fine_to_coarse': [coarse_index[key[wnid].name()] for wnid in wnids],
    }


def load(root):
    """Return the cached mapping payload, building and caching it if absent."""
    root = os.path.join(root, 'tiny-imagenet-200')
    path = os.path.join(root, CACHE_NAME)
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    payload = _build(root)
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=1)
    return payload


def fine_to_coarse_map(root):
    """Return a 200-length list: fine class id -> coarse (superclass) id."""
    return load(root)['fine_to_coarse']


def coarse_class_count(root):
    return len(load(root)['coarse_names'])


class CoarseLabelDataset:
    """Yield coarse labels from an ``ImageFolder``, keeping ``.targets`` on the fine ones.

    Assigning ``.transform`` detaches a shallow copy of the base first: the
    unlearning loaders build a clean-transform view with ``copy.copy``, and
    without the detach that assignment would also change the augmented view
    they were copied from.
    """

    def __init__(self, base, fine_to_coarse):
        self.base = base
        self.fine_to_coarse = list(fine_to_coarse)
        self.targets = list(base.targets)

    @property
    def transform(self):
        return self.base.transform

    @transform.setter
    def transform(self, value):
        self.base = copy.copy(self.base)
        self.base.transform = value

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, fine = self.base[index]
        return image, self.fine_to_coarse[fine]
