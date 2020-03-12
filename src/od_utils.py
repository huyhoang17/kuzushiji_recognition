import os
import cv2
import xml
import xml.etree.ElementTree as ET
import numpy as np

from utils import timer


@timer
def kuzu_gen_single_xml(img_fp,
                        labels,
                        save_xml_fd,
                        train_fd):
    """Gererate PASCAL format from dataset
    for object detection module
    """
    img_fn = img_fp.split("/")[-1].split(".")[0]
    img = cv2.imread(img_fp)[:, :, ::-1]

    try:
        labels = np.array(labels.split(' ')).reshape(-1, 5)
    except AttributeError:
        return None

    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = train_fd
    ET.SubElement(root, "filename").text = img_fn + ".jpg"
    ET.SubElement(root, "path").text = os.path.join(train_fd, img_fn) + ".jpg"

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(root, "size")
    height, width, depth = img.shape
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    ET.SubElement(root, "segmented").text = "0"

    for codepoint, x, y, w, h in labels:
        xmin, ymin, w, h = int(x), int(y), int(w), int(h)
        xmax = xmin + w
        ymax = ymin + h

        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = "kuzu"
        ET.SubElement(obj, "pose").text = 'Unspecified'
        ET.SubElement(obj, "truncated").text = '0'
        ET.SubElement(obj, "difficult").text = '0'

        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(xmin)
        ET.SubElement(bbox, "ymin").text = str(ymin)
        ET.SubElement(bbox, "xmax").text = str(xmax)
        ET.SubElement(bbox, "ymax").text = str(ymax)

    xml_str = ET.tostring(root).decode("utf-8")
    xml_indent = xml.dom.minidom.parseString(xml_str).toprettyxml(indent="\t")
    new_fn = "{}.xml".format(img_fn)
    new_fn = os.path.join(save_xml_fd, new_fn)
    with open(new_fn, "w") as f:
        f.write(xml_indent)
