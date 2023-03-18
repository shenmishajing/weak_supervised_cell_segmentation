import json
from collections import defaultdict


def main():
    data = json.load(
        open("data/livecell-dataset/annotations/LIVECell/livecell_coco_train.json")
    )

    image_shapes = defaultdict(int)
    for image in data["images"]:
        image_shapes[(image["width"], image["height"])] += 1

    for k, v in sorted(image_shapes.items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {v}")

    print("ann_per_image")

    ann_per_image = defaultdict(int)
    for ann in data["annotations"]:
        ann_per_image[ann["image_id"]] += 1

    res = sorted(ann_per_image.items(), key=lambda x: x[1], reverse=True)
    print(res[:10])
    print(res[-10:])


if __name__ == "__main__":
    main()
