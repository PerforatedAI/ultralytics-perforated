################################################################################
# Author:                                                                      #
# Nicholas Mesa-Cucalon (nicholas@perforatedai.com)                            #
#                                                                              #
# Download Cityscapes and build the 8-class detection set in YOLO layout.      #
################################################################################

#
"""
Imports
"""
import appdirs
import argparse
import cv2
import numpy as np
import subprocess
import zipfile

from multiprocessing import Pool, cpu_count
from pathlib         import Path
from typing          import Dict, List, Tuple

import cityscapesscripts.helpers.labels as cs_labels

from ultralytics.utils import LOGGER, TQDM

#
"""
Config
"""
# Cityscapes package names csDownload understands. gtFine carries the
# instance id maps the boxes come from, leftImg8bit carries the images.
download_packages = ('gtFine_trainvaltest.zip', 'leftImg8bit_trainvaltest.zip')

# csDownload reads its credentials from this file only
credentials_app       = ('cityscapesscripts', 'cityscapes')
credentials_file_name = 'credentials.json'

splits = ('train', 'val')

image_dir_name  = 'leftImg8bit'
label_dir_name  = 'gtFine'
image_suffix    = '_leftImg8bit.png'
instance_suffix = '_gtFine_instanceIds.png'

# Ids below this are stuff classes, so skipping them first is faster.
first_instance_id = 24

# An instance id under this value marks a crowd region rather than one
# object. mmdet tags those iscrowd and every COCO consumer drops them.
crowd_id_ceiling = 1000

# Boxes thinner than this in either axis are dropped, matching the guard
# convert_coco applies before writing a YOLO line.
min_box_pixels = 1


#
"""
Functions
"""
def build_class_map() -> Tuple[Dict[int, int], List[str]]:
    '''
    Map Cityscapes label ids onto contiguous YOLO class indices

    Notes:
        - YOLO label files need contiguous class indices, so the raw
          Cityscapes label ids are remapped here
        - hasInstances and not ignoreInEval is mmdet's filter for the
          eight detection classes
    '''
    selected = [
        label for label in cs_labels.labels
        if label.hasInstances and not label.ignoreInEval
    ]
    # Cityscapes label id mapped to its position in the class name list
    return (
        {label.id: index for index, label in enumerate(selected)},
        [label.name for label in selected],
    )


def build_label_lines(
    instance_path : Path,
    class_map     : Dict[int, int],
) -> List[str]:
    '''
    Turn one instance id map into normalized YOLO detection lines

    Notes:
        - Instance ids encode the class as id // 1000 above the crowd
          ceiling, and as the bare label id below it
        - The box is the tight extent of the instance mask, which is what
          pycocotools toBbox returns for the same mask in mmdet

    Signature:
        instance_path (Path):
            - Path to a gtFine instanceIds png
        class_map (Dict[int, int]):
            - Cityscapes label id mapped to contiguous class index
    '''
    instances = cv2.imread(str(instance_path), cv2.IMREAD_UNCHANGED)
    if instances is None:
        raise RuntimeError(f'Could not read instance map {instance_path}')
    height, width = instances.shape[:2]

    lines = []
    for instance_id in np.unique(instances[instances >= first_instance_id]):
        if instance_id < crowd_id_ceiling:
            continue
        label_id = int(instance_id) // crowd_id_ceiling
        class_index = class_map.get(label_id)
        if class_index is None:
            continue

        rows, cols = np.nonzero(instances == instance_id)
        box_width = cols.max() - cols.min() + 1
        box_height = rows.max() - rows.min() + 1
        if box_width < min_box_pixels or box_height < min_box_pixels:
            continue

        centre_x = (cols.min() + cols.max() + 1) / 2 / width
        centre_y = (rows.min() + rows.max() + 1) / 2 / height
        lines.append(
            f'{class_index} {centre_x:.6f} {centre_y:.6f} '
            f'{box_width / width:.6f} {box_height / height:.6f}'
        )
    return lines


def convert_one_image(job: Tuple[Path, Path, Path, Dict[int, int]]) -> int:
    '''
    Write one image's label file and link its image into place

    Notes:
        - ultralytics finds a label by swapping images for labels in
          the image path, so the two trees are siblings with equal stems

    Signature:
        job (Tuple[Path, Path, Path, Dict[int, int]]):
            - Instance map path, image path, split image dir, class map
    '''
    instance_path, image_path, image_dir, class_map = job
    label_dir = image_dir.parent.parent / 'labels' / image_dir.name

    # Keep the _leftImg8bit suffix, or the label file orphans its image
    label_path = label_dir / f'{image_path.stem}.txt'
    lines = build_label_lines(instance_path, class_map)
    label_path.write_text('\n'.join(lines) + ('\n' if lines else ''))

    target = image_dir / image_path.name
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(image_path.resolve())
    return len(lines)


def build_split(
    cityscapes_root : Path,
    out_root        : Path,
    split           : str,
    class_map       : Dict[int, int],
    workers         : int,
) -> Tuple[int, int]:
    '''
    Convert one split into YOLO labels and a flat image directory

    Notes:
        - The per city nesting is flattened, since the stems are unique

    Signature:
        cityscapes_root (Path):
            - Directory holding leftImg8bit and gtFine
        out_root (Path):
            - Destination dataset root
        split (str):
            - Split name, train or val
        class_map (Dict[int, int]):
            - Cityscapes label id mapped to contiguous class index
        workers (int):
            - Worker processes used for the conversion
    '''
    image_root = cityscapes_root / image_dir_name / split
    label_root = cityscapes_root / label_dir_name / split
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(
            f'Expected {image_root} and {label_root}. Run this script '
            f'with --download, or point --cityscapes-root at the '
            f'directory that holds {image_dir_name} and {label_dir_name}.'
        )

    image_out = out_root / 'images' / split
    label_out = out_root / 'labels' / split
    image_out.mkdir(parents = True, exist_ok = True)
    label_out.mkdir(parents = True, exist_ok = True)

    jobs = []
    for image_path in sorted(image_root.rglob(f'*{image_suffix}')):
        stem = image_path.name[: -len(image_suffix)]
        instance_path = (
            label_root / image_path.parent.name / f'{stem}{instance_suffix}'
        )
        if not instance_path.is_file():
            raise FileNotFoundError(
                f'Image {image_path} has no instance map at {instance_path}.'
            )
        jobs.append((instance_path, image_path, image_out, class_map))

    if not jobs:
        raise FileNotFoundError(
            f'Found no {image_suffix} files under {image_root}.'
        )

    with Pool(workers) as pool:
        counts = list(TQDM(
            pool.imap_unordered(convert_one_image, jobs, chunksize = 16),
            total = len(jobs),
            desc  = f'Converting {split}',
        ))
    return len(jobs), int(sum(counts))


def download_cityscapes(cityscapes_root: Path) -> None:
    '''
    Fetch and unzip the two Cityscapes packages this dataset needs

    Notes:
        - csDownload reads credentials.json and prompts when it is
          missing, so we check for it up front
        - An existing extraction is left alone, the images are 11 GB

    Signature:
        cityscapes_root (Path):
            - Directory the packages are downloaded and unzipped into
    '''
    credentials = (
        Path(appdirs.user_data_dir(*credentials_app)) / credentials_file_name
    )
    if not credentials.is_file():
        raise RuntimeError(
            f'Cityscapes downloads need an account, and csDownload reads it '
            f'only from {credentials}. Write that file as '
            f'{{"username": ..., "password": ...}} at 600 permissions, or '
            f'run csDownload once by hand and answer yes when it offers to '
            f'store the credentials.'
        )

    cityscapes_root.mkdir(parents = True, exist_ok = True)
    for package in download_packages:
        archive = cityscapes_root / package
        extracted = cityscapes_root / package.split('_')[0]
        if extracted.is_dir():
            LOGGER.info(f'{extracted} already exists, skipping {package}')
            continue
        if not archive.is_file():
            LOGGER.info(f'Downloading {package}')
            subprocess.run(
                ['csDownload', '-d', str(cityscapes_root), package],
                check = True,
            )
        LOGGER.info(f'Unzipping {package}')
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(cityscapes_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Build Cityscapes 8-class detection in YOLO layout'
    )
    parser.add_argument(
        '--cityscapes-root',
        type    = str,
        default = 'datasets/cityscapes-raw',
        help    = 'Directory holding leftImg8bit and gtFine, and where '
                  '--download puts them',
    )
    parser.add_argument(
        '--out-root',
        type    = str,
        default = 'datasets/cityscapes-det',
        help    = 'Destination dataset root, which must match the path '
                  'key in cfg/cityscapes-det.yaml',
    )
    parser.add_argument(
        '--download',
        action  = 'store_true',
        default = False,
        help    = 'Fetch the Cityscapes packages before converting',
    )
    parser.add_argument(
        '--workers',
        type    = int,
        default = max(1, cpu_count() - 1),
        help    = 'Worker processes used for the conversion',
    )
    args = parser.parse_args()

    cityscapes_root = Path(args.cityscapes_root).expanduser()
    out_root = Path(args.out_root).expanduser()

    if args.download:
        download_cityscapes(cityscapes_root)

    # Build Class Map
    class_map, class_names = build_class_map()
    LOGGER.info(f'Class order: {class_names}')

    # Convert Splits
    for split_name in splits:
        images, boxes = build_split(
            cityscapes_root,
            out_root,
            split_name,
            class_map,
            args.workers,
        )
        LOGGER.info(f'{split_name}: {images} images, {boxes} boxes')

    LOGGER.info(
        f'Cityscapes detection set ready at {out_root}. Confirm the names '
        f'order in cfg/cityscapes-det.yaml matches the class order above.'
    )
