import os

from detectron2.data.datasets import register_coco_instances

# True for open source;
# Internally at fb, we register them elsewhere
# Assume pre-defined datasets live in `./datasets`.
root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))

for split, image_root in zip(
    ["train", "val", "test"], ["train_val", "train_val", "test"]
):
    register_coco_instances(
        f"livecell_{split}",
        {},
        os.path.join(
            root,
            f"livecell-dataset/annotations/LIVECell/livecell_coco_{split}.json",
        ),
        os.path.join(root, f"livecell-dataset/images/livecell_{image_root}_images"),
    )

for split, json_file, image_root in zip(
    ["train", "val"],
    ["train", "test"],
    ["train", "tt"],
):
    register_coco_instances(
        f"coco_dna_{split}",
        {},
        os.path.join(
            root,
            f"coco_dna/annotations/{json_file}.json",
        ),
        os.path.join(root, f"coco_dna/{image_root}"),
    )
