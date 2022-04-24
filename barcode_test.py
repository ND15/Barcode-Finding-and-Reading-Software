# Libraries

from imutils import paths
import os
import cv2
import argparse
from pyzbar.pyzbar import decode
import numpy as np
from collections import Counter

scale = 0.3


def barcode_decode(i, detectedBarcodes, img_p):
    """
    :param i: index of the input image
    :param detectedBarcodes: return value of the decode function of pyzbar lib
    :param img_p: processed image

    ** Creates and displays blue bounding box around the barcodes read correctly **
    """
    li = []

    print("For image " + str(i) + " the barcodes read are: ")
    if not detectedBarcodes:
        print("Barcode Not Detected or your barcode is blank/corrupted!")

    else:
        for barcode in detectedBarcodes:
            bData = barcode.data.decode('utf-8')
            if barcode.type == "QRCODE":
                continue
            li.append(bData)
            pts = np.array([barcode.polygon], np.int32)
            pts = pts.reshape((-1, 1, 2))
            pts = np.int32(pts * scale)
            cv2.polylines(img_p, [pts], True, (255, 0, 0), 5)
        print(Counter(li))

    cv2.imshow("Barcode Bound box", img_p)
    cv2.waitKey(0)


def read_input_and_barcode(image_paths):
    """

    :param image_paths: path of the input images directory
    ** Displays input images and barcode bounding box **
    """
    image_paths = list(paths.list_images(args["dataset"]))
    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        img_p = img

        width = int(img_p.shape[1] * scale)
        height = int(img_p.shape[0] * scale)
        img_p = cv2.resize(img_p, (width, height))
        cv2.imshow("Input Image " + str(i), img_p)
        cv2.waitKey(0)

        detectedBarcodes = decode(img)
        barcode_decode(i, detectedBarcodes, img_p)
        cv2.destroyAllWindows()


def bound_items(image_paths):
    for i, image_path in enumerate(image_paths):
        print("entered")
        img = cv2.imread(image_path)
        img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
        classNames = []
        classFile = 'coco.names'
        with open(classFile, "rt") as f:
            classNames = f.read().rstrip('\n').split('\n')

        configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightsPath = 'frozen_inference_graph.pb'

        net = cv2.dnn_DetectionModel(weightsPath, configPath)
        net.setInputSize(480, 480)
        net.setInputScale(2.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)

        classIds, confs, bbox = net.detect(img, confThreshold=0.45, nmsThreshold=0.1)

        i = 0
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.imwrite("bound" + str(i) + ".jpg", img[box[1]: box[1] + box[3], box[0]: box[0] + box[2]])
                cv2.rectangle(img, box, (0, 0, 0), 2)
                cv2.putText(img, "Item", (box[0] + 10, box[1] + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                # box = np.reshape(box, (2, 2))
                print(box)
                i += 1

        cv2.imshow("out", img)
        cv2.imwrite("img_" + str(i), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input folder")
    args = vars(ap.parse_args())
    image_paths = list(paths.list_images(args["dataset"]))

    # Outputs the value of every barcode and how many times each barcode appears
    # display each input image and creates the blue bounding boxes for barcodes read
    # correctly
    read_input_and_barcode(image_paths)

    # Creates bounding box of black color around each input item
    bound_items(image_paths)
