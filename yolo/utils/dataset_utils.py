import json
import os
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from yolo.tools.data_conversion import discretize_categories


def locate_label_paths(dataset_path: Path, phase_name: Path) -> Union[Tuple[Path, str], Tuple[None, None]]:
    """
    Find the path to label files for a specified dataset and phase(e.g. training).

    Args:
        dataset_path (Path): The path to the root directory of the dataset.
        phase_name (Path): The name of the phase for which labels are being searched (e.g., "train", "val", "test").

    Returns:
        Tuple[Path, Path]: A tuple containing the path to the labels file and the file format ("json" or "txt").
    """
    json_labels_path = dataset_path / "annotations" / f"instances_{phase_name}.json"

    txt_labels_path = dataset_path / "labels" / phase_name

    if json_labels_path.is_file():
        return json_labels_path, "json"

    elif txt_labels_path.is_dir():
        txt_files = [f for f in os.listdir(txt_labels_path) if f.endswith(".txt")]
        if txt_files:
            return txt_labels_path, "txt"

    logger.warning("No labels found in the specified dataset path and phase name.")
    return None, None


def create_image_metadata(labels_path: str) -> Tuple[Dict[str, List], Dict[str, Dict]]:
    """
    Create a dictionary containing image information and annotations indexed by image ID.

    Args:
        labels_path (str): The path to the annotation json file.

    Returns:
        - annotations_index: A dictionary where keys are image IDs and values are lists of annotations.
        - image_info_dict: A dictionary where keys are image file names without extension and values are image information dictionaries.
    """
    with open(labels_path, "r") as file:
        labels_data = json.load(file)
        id_to_idx = discretize_categories(labels_data.get("categories", [])) if "categories" in labels_data else None
        annotations_index = organize_annotations_by_image(labels_data, id_to_idx)  # check lookup is a good name?
        image_info_dict = {Path(img["file_name"]).stem: img for img in labels_data["images"]}
        return annotations_index, image_info_dict


def organize_annotations_by_image(data: Dict[str, Any], id_to_idx: Optional[Dict[int, int]]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Use image index to lookup every annotations
    Args:
        data (Dict[str, Any]): A dictionary containing annotation data.

    Returns:
        Dict[int, List[Dict[str, Any]]]: A dictionary where keys are image IDs and values are lists of annotations.
        Annotations with "iscrowd" set to True are excluded from the index.

    """
    annotation_lookup = {}
    for anno in data["annotations"]:
        if anno["iscrowd"]:
            continue
        image_id = anno["image_id"]
        if id_to_idx:
            anno["category_id"] = id_to_idx[anno["category_id"]]
        if image_id not in annotation_lookup:
            annotation_lookup[image_id] = []
        annotation_lookup[image_id].append(anno)
    return annotation_lookup


def scale_segmentation(
    annotations: List[Dict[str, Any]], image_dimensions: Dict[str, int]
) -> Optional[List[List[float]]]:
    """
    Scale the segmentation data based on image dimensions and return a list of scaled segmentation data.

    Args:
        annotations (List[Dict[str, Any]]): A list of annotation dictionaries.
        image_dimensions (Dict[str, int]): A dictionary containing image dimensions (height and width).

    Returns:
        Optional[List[List[float]]]: A list of scaled segmentation data, where each sublist contains category_id followed by scaled (x, y) coordinates.
    """
    if annotations is None:
        return None

    seg_array_with_cat = []
    h, w = image_dimensions["height"], image_dimensions["width"]
    for anno in annotations:
        category_id = anno["category_id"]
        if "segmentation" in anno:
            seg_list = [item for sublist in anno["segmentation"] for item in sublist]
        elif "bbox" in anno:
            seg_list = anno["bbox"]
        scaled_seg_data = (
            np.array(seg_list).reshape(-1, 2) / [w, h]
        ).tolist()  # make the list group in x, y pairs and scaled with image width, height
        scaled_flat_seg_data = [category_id] + list(chain(*scaled_seg_data))  # flatten the scaled_seg_data list
        seg_array_with_cat.append(scaled_flat_seg_data)

    return seg_array_with_cat
