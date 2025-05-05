# Global imports
from PIL import Image
import numpy as np
import albumentations as albu
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import copy
import tqdm


# Get dict of annotations usable for visualization
def get_anno_dict(coco_path, coco_dict=None):
    # Load coco dict
    if coco_dict is None:
        with open(coco_path, 'r') as fp:
            coco_dict = json.load(fp)
        
    # Build image dict
    anno_dict = {}
    for image_anno in coco_dict['images']:
        # copy the anno so original doesn't change
        anno = copy.deepcopy(image_anno)
        image_id = image_anno['id']
        # add new info
        anno['bboxes'] = []
        anno['person_ids'] = []
        anno['is_known'] = []
        # store the anno
        anno_dict[image_id] = anno
        
    # Build dictionary of images with their bounding box annotations
    for anno in coco_dict['annotations']:
        image_id = anno['image_id']
        anno_dict[image_id]['bboxes'].append(anno['bbox'])
        anno_dict[image_id]['person_ids'].append(anno['person_id'])
        anno_dict[image_id]['is_known'].append(anno['is_known'])
        
    # Return anno dict
    return anno_dict

# Load image
def load_image(image_dir, image_anno):
    image_file = image_anno['file_name']
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path)
    arr = np.array(image)
    loaded_anno = copy.deepcopy(image_anno)
    loaded_anno['image'] = arr
    return loaded_anno

# Show image
def show_image(loaded_anno):
    plt.imshow(loaded_anno['image'])
    plt.show()

# Transform function
def transform_anno(transform, loaded_anno):
    image = loaded_anno['image']
    bboxes = loaded_anno['bboxes']
    person_ids = loaded_anno['person_ids']
    is_known = loaded_anno['is_known']
    transformed_anno = transform(image=image, bboxes=bboxes, person_ids=person_ids, is_known=is_known)
    return transformed_anno

# Plot detected boxes on image with matplotlib
def show_detects(anno, ax=None, title=None, xlabel=None, figsize=(6, 6), remove_ticks=True):
    # Setup subplot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    # Setup labels
    if title is not None:
        ax.set_title(title, fontsize=20, fontweight='bold')
    # Show the image
    ax.imshow(anno['image'])
    # Plot boxes (and optionally similarity scores)
    for i, (box, pid) in enumerate(zip(anno['bboxes'], anno['person_ids'])):
        x, y, w, h = box
        ax.add_patch(Rectangle((x, y), w, h, edgecolor='black', lw=2, fill=False, alpha=0.8))
        ax.add_patch(Rectangle((x+1, y+1), w-2, h-2, edgecolor='yellow', lw=1, fill=False, alpha=0.8))
        ## Plot label
        ax.text(x, y, pid, ha="left", va="bottom", size=10,
            bbox=dict(boxstyle="square,pad=0.15", fc="whitesmoke", alpha=0.8, ec='black', lw=1.0)
        )
    # Remove ticks and expand borders
    if remove_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    [x.set_linewidth(2) for x in ax.spines.values()]

# Show detects for multiple images next to each other
def show_multi(image_list, plot_width=6, plot_height=6, preserve_relative_scale=True):
    n = len(image_list)
    if preserve_relative_scale:
        height_list = [image['image'].shape[0] for image in image_list]
        width_list = [image['image'].shape[1] for image in image_list]
        max_height = max(height_list)
        width_ratios = [w/sum(width_list) for w in width_list]
        _fig_width = sum(width_list) / 100
        _plot_height = max_height / 100
    else:
        _fig_width = plot_width * n
        _plot_height = plot_height
        width_ratios = None
    #
    fig, ax_arr = plt.subplots(nrows=1, ncols=n, figsize=(_fig_width, _plot_height), width_ratios=width_ratios)
    for ax, image in zip(ax_arr, image_list):
        show_detects(image, ax=ax, remove_ticks=not preserve_relative_scale)
    if preserve_relative_scale:
        plt.subplots_adjust(wspace=0, hspace=0)
