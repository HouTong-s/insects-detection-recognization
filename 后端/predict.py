from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2 as cv
import numpy as np
from torchvision import transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import connector


def check_box(boxes):
    overlap_idx = []
    for i in range(len(boxes)):
        box1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            box2 = boxes[j]
            if box1[0] > box2[2] or box1[1] > box2[3] or box1[2] < box2[0] or box1[3] < box2[1]:
                continue
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            overlap_area = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            total_area = area1 + area2 - overlap_area
            if 2 * overlap_area > total_area or overlap_area > 0.9 * area2 or overlap_area > 0.9 * area1:
                if area1 > area2:
                    overlap_idx.append((i, j))
                else:
                    overlap_idx.append((j, i))

    return overlap_idx


class Predictor:
    def __init__(self):
        self.orders = ["Araneae", "Coleoptera", "Diptera", "Hemiptera", "Hymenoptera", "Lepidoptera", "Odonata"]
        self.test_transform = A.Compose([
            ToTensorV2(p=1.0)
        ])
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.detector, self.classifier = self.load_models()
        self.classify_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # 将PIL图像转为Tensor，并且进行归一化
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
        ])
        self.species = ['t100', 't101', 't16', 't33', 't34', 't40', 't42', 't54', 't75', 't8', 't84', 't85', 't88',
                        't93', 't95', 't97']

    def load_models(self):
        # detect
        detect_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
        num_order_classes = len(self.orders)
        in_features = detect_model.roi_heads.box_predictor.cls_score.in_features
        detect_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_order_classes)
        detect_model.load_state_dict(torch.load("fasterrcnn_resnet50_fpn_6.pth", map_location=self.device))
        detect_model.eval()

        # classify
        classifier = torchvision.models.resnext50_32x4d(pretrained=False, progress=False)
        num_fc_in = classifier.fc.in_features
        classifier.fc = torch.nn.Linear(num_fc_in, 16)
        classifier.load_state_dict(torch.load("resnet50.pt", map_location=torch.device(self.device)))
        classifier.eval()
        return detect_model, classifier

    def predict(self, dic_name, img_name, write_path):
        image = cv.imread(dic_name + "/" + img_name, cv.IMREAD_COLOR)
        # print(image.shape)
        height, _, _ = image.shape
        img = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32)
        img = img / 255.0
        sample = {
            'image': img,
        }
        sample = self.test_transform(**sample)
        img = sample["image"]

        output = self.detector([img])

        output = output[0]
        boxes = output["boxes"].tolist()
        labels = output["labels"].tolist()
        scores = output["scores"].tolist()

        final_boxes = []
        final_order_labels = []
        final_result = {"order_labels": [], "family_labels": [], "genus_labels": [], "species_labels": [], "boxes": [],
                        "picture": []}
        try:
            qualified_box = 0
            for i in range(len(scores)):
                # if scores[i] > 0.2:
                qualified_box = qualified_box + 1
                box = list(map(int, boxes[i]))
                final_boxes.append(box)
                final_order_labels.append(labels[i])
                # cv.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 5)
            # if qualified_box == 0:
            #     max_index = scores.index(max(scores))
            #     box = boxes[max_index]
            #     cv.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 5)
            temp_overlaps = check_box(final_boxes)
            overlaps = []
            for left_laps, right_laps in temp_overlaps:
                if scores[left_laps] > 0.5 and scores[right_laps] <= 0.5:
                    overlaps.append(right_laps)
                elif scores[left_laps] <= 0.5 and scores[right_laps] > 0.5:
                    overlaps.append(left_laps)
                else:
                    overlaps.append(right_laps)

            overlaps = np.unique(np.array(overlaps)).tolist()

            final_result["boxes"] = [final_boxes[i] for i in range(0, len(final_boxes)) if i not in overlaps]
            # final_result["order_labels"] = [final_order_labels[i] for i in range(0, len(final_order_labels)) if
            #                                 i not in overlaps]

            cla_slices = np.array([])

            for box in final_result["boxes"]:
                cla_slice_cv = image[box[1]:box[3], box[0]:box[2]]
                cla_slice = Image.fromarray(cv.cvtColor(cla_slice_cv, cv.COLOR_BGR2RGB))
                cla_slice = self.classify_transform(cla_slice)
                cla_slices = np.append(cla_slices, cla_slice.numpy())

            cla_slices = cla_slices.reshape(-1, 3, 224, 224)
            cla_slices = torch.from_numpy(cla_slices).float()

            class_results = self.classifier(cla_slices)
            _, predicted = torch.max(class_results.data, 1)

            i = 0
            for pre in predicted.tolist():
                name = self.species[pre]
                name = name.split("t")[1]
                path = "/" + name
                _, mu, shu, ke, zhong, _ = connector.fetch_specie(path)
                final_result["order_labels"].append(mu)
                final_result["family_labels"].append(ke)
                final_result["genus_labels"].append(shu)
                final_result["species_labels"].append(zhong)
                i = i + 1

            i = 0
            for box in final_result["boxes"]:
                cv.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 5)
                cv.putText(image, str(i), (int(box[0]) + 5, int(box[1]) + 5), cv.FONT_HERSHEY_SIMPLEX, 1.25 * height / 394,
                           (255, 0, 0), int(1.25 * height / 394) + 1)
                i = i + 1
        except RuntimeError:
            pass
        finally:
            final_result["picture"] = "http://139.224.50.124:8081/result/" + img_name + "_answer.png"
            cv.imwrite(write_path + "/" + img_name + "_answer.png", image)

        return final_result



