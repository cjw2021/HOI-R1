import os
import json

import numpy as np
from tqdm import tqdm

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import torch.multiprocessing as mp
from datetime import datetime
import argparse

from postprocess import extract_and_validate_hoi_output

# Example:
# python eval_qwen3.py \
#     --model_path Qwen/Qwen3-VL-4B-Instruct \
#     --test_anno /path/to/HICO_Det/annotations/test_hico.json \
#     --image_folder /path/to/HICO_Det/images/test2015 \
#     --output_folder ./evaluate_predictions/qwen3-vl-4b \
#     --num_gpus 8 --section 0 --max_section 1


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model on HICO-DET dataset")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-4B-Instruct", help="Path to the model checkpoint")
    parser.add_argument("--test_anno", type=str, required=True, help="Path to HICO-DET test_hico.json annotation file")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to HICO-DET test2015 image folder")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save evaluation predictions")
    parser.add_argument("--existing_merged_json", type=str, default="", help="Path to existing merged JSON file")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs for multiprocessing")
    parser.add_argument("--section", type=int, default=0, help="Current section index (0-based)")
    parser.add_argument("--max_section", type=int, default=1, help="Total number of sections to divide the dataset")
    return parser.parse_args()

hico_text_label = {(4, 4): 'a photo of a person boarding an airplane',
                   (17, 4): 'a photo of a person directing an airplane',
                   (25, 4): 'a photo of a person exiting an airplane',
                   (30, 4): 'a photo of a person flying an airplane',
                   (41, 4): 'a photo of a person inspecting an airplane',
                   (52, 4): 'a photo of a person loading an airplane',
                   (76, 4): 'a photo of a person riding an airplane',
                   (87, 4): 'a photo of a person sitting on an airplane',
                   (111, 4): 'a photo of a person washing an airplane',
                   (57, 4): 'a photo of a person and an airplane', (8, 1): 'a photo of a person carrying a bicycle',
                   (36, 1): 'a photo of a person holding a bicycle',
                   (41, 1): 'a photo of a person inspecting a bicycle',
                   (43, 1): 'a photo of a person jumping a bicycle',
                   (37, 1): 'a photo of a person hopping on a bicycle',
                   (62, 1): 'a photo of a person parking a bicycle',
                   (71, 1): 'a photo of a person pushing a bicycle',
                   (75, 1): 'a photo of a person repairing a bicycle',
                   (76, 1): 'a photo of a person riding a bicycle',
                   (87, 1): 'a photo of a person sitting on a bicycle',
                   (98, 1): 'a photo of a person straddling a bicycle',
                   (110, 1): 'a photo of a person walking a bicycle',
                   (111, 1): 'a photo of a person washing a bicycle', (57, 1): 'a photo of a person and a bicycle',
                   (10, 14): 'a photo of a person chasing a bird', (26, 14): 'a photo of a person feeding a bird',
                   (36, 14): 'a photo of a person holding a bird', (65, 14): 'a photo of a person petting a bird',
                   (74, 14): 'a photo of a person releasing a bird',
                   (112, 14): 'a photo of a person watching a bird', (57, 14): 'a photo of a person and a bird',
                   (4, 8): 'a photo of a person boarding a boat', (21, 8): 'a photo of a person driving a boat',
                   (25, 8): 'a photo of a person exiting a boat', (41, 8): 'a photo of a person inspecting a boat',
                   (43, 8): 'a photo of a person jumping a boat', (47, 8): 'a photo of a person launching a boat',
                   (75, 8): 'a photo of a person repairing a boat', (76, 8): 'a photo of a person riding a boat',
                   (77, 8): 'a photo of a person rowing a boat', (79, 8): 'a photo of a person sailing a boat',
                   (87, 8): 'a photo of a person sitting on a boat',
                   (93, 8): 'a photo of a person standing on a boat', (105, 8): 'a photo of a person tying a boat',
                   (111, 8): 'a photo of a person washing a boat', (57, 8): 'a photo of a person and a boat',
                   (8, 39): 'a photo of a person carrying a bottle',
                   (20, 39): 'a photo of a person drinking with a bottle',
                   (36, 39): 'a photo of a person holding a bottle',
                   (41, 39): 'a photo of a person inspecting a bottle',
                   (48, 39): 'a photo of a person licking a bottle',
                   (58, 39): 'a photo of a person opening a bottle',
                   (69, 39): 'a photo of a person pouring a bottle', (57, 39): 'a photo of a person and a bottle',
                   (4, 5): 'a photo of a person boarding a bus', (17, 5): 'a photo of a person directing a bus',
                   (21, 5): 'a photo of a person driving a bus', (25, 5): 'a photo of a person exiting a bus',
                   (41, 5): 'a photo of a person inspecting a bus', (52, 5): 'a photo of a person loading a bus',
                   (76, 5): 'a photo of a person riding a bus', (87, 5): 'a photo of a person sitting on a bus',
                   (111, 5): 'a photo of a person washing a bus', (113, 5): 'a photo of a person waving a bus',
                   (57, 5): 'a photo of a person and a bus', (4, 2): 'a photo of a person boarding a car',
                   (17, 2): 'a photo of a person directing a car', (21, 2): 'a photo of a person driving a car',
                   (38, 2): 'a photo of a person hosing a car', (41, 2): 'a photo of a person inspecting a car',
                   (43, 2): 'a photo of a person jumping a car', (52, 2): 'a photo of a person loading a car',
                   (62, 2): 'a photo of a person parking a car', (76, 2): 'a photo of a person riding a car',
                   (111, 2): 'a photo of a person washing a car', (57, 2): 'a photo of a person and a car',
                   (22, 15): 'a photo of a person drying a cat', (26, 15): 'a photo of a person feeding a cat',
                   (36, 15): 'a photo of a person holding a cat', (39, 15): 'a photo of a person hugging a cat',
                   (45, 15): 'a photo of a person kissing a cat', (65, 15): 'a photo of a person petting a cat',
                   (80, 15): 'a photo of a person scratching a cat', (111, 15): 'a photo of a person washing a cat',
                   (10, 15): 'a photo of a person chasing a cat', (57, 15): 'a photo of a person and a cat',
                   (8, 56): 'a photo of a person carrying a chair', (36, 56): 'a photo of a person holding a chair',
                   (49, 56): 'a photo of a person lying on a chair',
                   (87, 56): 'a photo of a person sitting on a chair',
                   (93, 56): 'a photo of a person standing on a chair', (57, 56): 'a photo of a person and a chair',
                   (8, 57): 'a photo of a person carrying a couch',
                   (49, 57): 'a photo of a person lying on a couch',
                   (87, 57): 'a photo of a person sitting on a couch', (57, 57): 'a photo of a person and a couch',
                   (26, 19): 'a photo of a person feeding a cow', (34, 19): 'a photo of a person herding a cow',
                   (36, 19): 'a photo of a person holding a cow', (39, 19): 'a photo of a person hugging a cow',
                   (45, 19): 'a photo of a person kissing a cow', (46, 19): 'a photo of a person lassoing a cow',
                   (55, 19): 'a photo of a person milking a cow', (65, 19): 'a photo of a person petting a cow',
                   (76, 19): 'a photo of a person riding a cow', (110, 19): 'a photo of a person walking a cow',
                   (57, 19): 'a photo of a person and a cow',
                   (12, 60): 'a photo of a person cleaning a dining table',
                   (24, 60): 'a photo of a person eating at a dining table',
                   (86, 60): 'a photo of a person sitting at a dining table',
                   (57, 60): 'a photo of a person and a dining table',
                   (8, 16): 'a photo of a person carrying a dog', (22, 16): 'a photo of a person drying a dog',
                   (26, 16): 'a photo of a person feeding a dog', (33, 16): 'a photo of a person grooming a dog',
                   (36, 16): 'a photo of a person holding a dog', (38, 16): 'a photo of a person hosing a dog',
                   (39, 16): 'a photo of a person hugging a dog', (41, 16): 'a photo of a person inspecting a dog',
                   (45, 16): 'a photo of a person kissing a dog', (65, 16): 'a photo of a person petting a dog',
                   (78, 16): 'a photo of a person running a dog', (80, 16): 'a photo of a person scratching a dog',
                   (98, 16): 'a photo of a person straddling a dog',
                   (107, 16): 'a photo of a person training a dog', (110, 16): 'a photo of a person walking a dog',
                   (111, 16): 'a photo of a person washing a dog', (10, 16): 'a photo of a person chasing a dog',
                   (57, 16): 'a photo of a person and a dog', (26, 17): 'a photo of a person feeding a horse',
                   (33, 17): 'a photo of a person grooming a horse',
                   (36, 17): 'a photo of a person holding a horse', (39, 17): 'a photo of a person hugging a horse',
                   (43, 17): 'a photo of a person jumping a horse', (45, 17): 'a photo of a person kissing a horse',
                   (52, 17): 'a photo of a person loading a horse',
                   (37, 17): 'a photo of a person hopping on a horse',
                   (65, 17): 'a photo of a person petting a horse', (72, 17): 'a photo of a person racing a horse',
                   (76, 17): 'a photo of a person riding a horse', (78, 17): 'a photo of a person running a horse',
                   (98, 17): 'a photo of a person straddling a horse',
                   (107, 17): 'a photo of a person training a horse',
                   (110, 17): 'a photo of a person walking a horse',
                   (111, 17): 'a photo of a person washing a horse', (57, 17): 'a photo of a person and a horse',
                   (36, 3): 'a photo of a person holding a motorcycle',
                   (41, 3): 'a photo of a person inspecting a motorcycle',
                   (43, 3): 'a photo of a person jumping a motorcycle',
                   (37, 3): 'a photo of a person hopping on a motorcycle',
                   (62, 3): 'a photo of a person parking a motorcycle',
                   (71, 3): 'a photo of a person pushing a motorcycle',
                   (72, 3): 'a photo of a person racing a motorcycle',
                   (76, 3): 'a photo of a person riding a motorcycle',
                   (87, 3): 'a photo of a person sitting on a motorcycle',
                   (98, 3): 'a photo of a person straddling a motorcycle',
                   (108, 3): 'a photo of a person turning a motorcycle',
                   (110, 3): 'a photo of a person walking a motorcycle',
                   (111, 3): 'a photo of a person washing a motorcycle',
                   (57, 3): 'a photo of a person and a motorcycle', (8, 0): 'a photo of a person carrying a person',
                   (31, 0): 'a photo of a person greeting a person',
                   (36, 0): 'a photo of a person holding a person', (39, 0): 'a photo of a person hugging a person',
                   (45, 0): 'a photo of a person kissing a person',
                   (92, 0): 'a photo of a person stabbing a person',
                   (100, 0): 'a photo of a person tagging a person',
                   (102, 0): 'a photo of a person teaching a person',
                   (48, 0): 'a photo of a person licking a person', (57, 0): 'a photo of a person and a person',
                   (8, 58): 'a photo of a person carrying a potted plant',
                   (36, 58): 'a photo of a person holding a potted plant',
                   (38, 58): 'a photo of a person hosing a potted plant',
                   (57, 58): 'a photo of a person and a potted plant',
                   (8, 18): 'a photo of a person carrying a sheep', (26, 18): 'a photo of a person feeding a sheep',
                   (34, 18): 'a photo of a person herding a sheep', (36, 18): 'a photo of a person holding a sheep',
                   (39, 18): 'a photo of a person hugging a sheep', (45, 18): 'a photo of a person kissing a sheep',
                   (65, 18): 'a photo of a person petting a sheep', (76, 18): 'a photo of a person riding a sheep',
                   (83, 18): 'a photo of a person shearing a sheep',
                   (110, 18): 'a photo of a person walking a sheep',
                   (111, 18): 'a photo of a person washing a sheep', (57, 18): 'a photo of a person and a sheep',
                   (4, 6): 'a photo of a person boarding a train', (21, 6): 'a photo of a person driving a train',
                   (25, 6): 'a photo of a person exiting a train', (52, 6): 'a photo of a person loading a train',
                   (76, 6): 'a photo of a person riding a train', (87, 6): 'a photo of a person sitting on a train',
                   (111, 6): 'a photo of a person washing a train', (57, 6): 'a photo of a person and a train',
                   (13, 62): 'a photo of a person controlling a tv', (75, 62): 'a photo of a person repairing a tv',
                   (112, 62): 'a photo of a person watching a tv', (57, 62): 'a photo of a person and a tv',
                   (7, 47): 'a photo of a person buying an apple', (15, 47): 'a photo of a person cutting an apple',
                   (23, 47): 'a photo of a person eating an apple',
                   (36, 47): 'a photo of a person holding an apple',
                   (41, 47): 'a photo of a person inspecting an apple',
                   (64, 47): 'a photo of a person peeling an apple',
                   (66, 47): 'a photo of a person picking an apple',
                   (89, 47): 'a photo of a person smelling an apple',
                   (111, 47): 'a photo of a person washing an apple', (57, 47): 'a photo of a person and an apple',
                   (8, 24): 'a photo of a person carrying a backpack',
                   (36, 24): 'a photo of a person holding a backpack',
                   (41, 24): 'a photo of a person inspecting a backpack',
                   (58, 24): 'a photo of a person opening a backpack',
                   (114, 24): 'a photo of a person wearing a backpack',
                   (57, 24): 'a photo of a person and a backpack', (7, 46): 'a photo of a person buying a banana',
                   (8, 46): 'a photo of a person carrying a banana',
                   (15, 46): 'a photo of a person cutting a banana',
                   (23, 46): 'a photo of a person eating a banana',
                   (36, 46): 'a photo of a person holding a banana',
                   (41, 46): 'a photo of a person inspecting a banana',
                   (64, 46): 'a photo of a person peeling a banana',
                   (66, 46): 'a photo of a person picking a banana',
                   (89, 46): 'a photo of a person smelling a banana', (57, 46): 'a photo of a person and a banana',
                   (5, 34): 'a photo of a person breaking a baseball bat',
                   (8, 34): 'a photo of a person carrying a baseball bat',
                   (36, 34): 'a photo of a person holding a baseball bat',
                   (84, 34): 'a photo of a person signing a baseball bat',
                   (99, 34): 'a photo of a person swinging a baseball bat',
                   (104, 34): 'a photo of a person throwing a baseball bat',
                   (115, 34): 'a photo of a person wielding a baseball bat',
                   (57, 34): 'a photo of a person and a baseball bat',
                   (36, 35): 'a photo of a person holding a baseball glove',
                   (114, 35): 'a photo of a person wearing a baseball glove',
                   (57, 35): 'a photo of a person and a baseball glove',
                   (26, 21): 'a photo of a person feeding a bear', (40, 21): 'a photo of a person hunting a bear',
                   (112, 21): 'a photo of a person watching a bear', (57, 21): 'a photo of a person and a bear',
                   (12, 59): 'a photo of a person cleaning a bed', (49, 59): 'a photo of a person lying on a bed',
                   (87, 59): 'a photo of a person sitting on a bed', (57, 59): 'a photo of a person and a bed',
                   (41, 13): 'a photo of a person inspecting a bench',
                   (49, 13): 'a photo of a person lying on a bench',
                   (87, 13): 'a photo of a person sitting on a bench', (57, 13): 'a photo of a person and a bench',
                   (8, 73): 'a photo of a person carrying a book', (36, 73): 'a photo of a person holding a book',
                   (58, 73): 'a photo of a person opening a book', (73, 73): 'a photo of a person reading a book',
                   (57, 73): 'a photo of a person and a book', (36, 45): 'a photo of a person holding a bowl',
                   (96, 45): 'a photo of a person stirring a bowl', (111, 45): 'a photo of a person washing a bowl',
                   (48, 45): 'a photo of a person licking a bowl', (57, 45): 'a photo of a person and a bowl',
                   (15, 50): 'a photo of a person cutting a broccoli',
                   (23, 50): 'a photo of a person eating a broccoli',
                   (36, 50): 'a photo of a person holding a broccoli',
                   (89, 50): 'a photo of a person smelling a broccoli',
                   (96, 50): 'a photo of a person stirring a broccoli',
                   (111, 50): 'a photo of a person washing a broccoli',
                   (57, 50): 'a photo of a person and a broccoli', (3, 55): 'a photo of a person blowing a cake',
                   (8, 55): 'a photo of a person carrying a cake', (15, 55): 'a photo of a person cutting a cake',
                   (23, 55): 'a photo of a person eating a cake', (36, 55): 'a photo of a person holding a cake',
                   (51, 55): 'a photo of a person lighting a cake', (54, 55): 'a photo of a person making a cake',
                   (67, 55): 'a photo of a person picking up a cake', (57, 55): 'a photo of a person and a cake',
                   (8, 51): 'a photo of a person carrying a carrot',
                   (14, 51): 'a photo of a person cooking a carrot',
                   (15, 51): 'a photo of a person cutting a carrot',
                   (23, 51): 'a photo of a person eating a carrot',
                   (36, 51): 'a photo of a person holding a carrot',
                   (64, 51): 'a photo of a person peeling a carrot',
                   (89, 51): 'a photo of a person smelling a carrot',
                   (96, 51): 'a photo of a person stirring a carrot',
                   (111, 51): 'a photo of a person washing a carrot', (57, 51): 'a photo of a person and a carrot',
                   (8, 67): 'a photo of a person carrying a cell phone',
                   (36, 67): 'a photo of a person holding a cell phone',
                   (73, 67): 'a photo of a person reading a cell phone',
                   (75, 67): 'a photo of a person repairing a cell phone',
                   (101, 67): 'a photo of a person talking on a cell phone',
                   (103, 67): 'a photo of a person texting on a cell phone',
                   (57, 67): 'a photo of a person and a cell phone',
                   (11, 74): 'a photo of a person checking a clock',
                   (36, 74): 'a photo of a person holding a clock',
                   (75, 74): 'a photo of a person repairing a clock',
                   (82, 74): 'a photo of a person setting a clock', (57, 74): 'a photo of a person and a clock',
                   (8, 41): 'a photo of a person carrying a cup',
                   (20, 41): 'a photo of a person drinking with a cup',
                   (36, 41): 'a photo of a person holding a cup', (41, 41): 'a photo of a person inspecting a cup',
                   (69, 41): 'a photo of a person pouring a cup', (85, 41): 'a photo of a person sipping a cup',
                   (89, 41): 'a photo of a person smelling a cup', (27, 41): 'a photo of a person filling a cup',
                   (111, 41): 'a photo of a person washing a cup', (57, 41): 'a photo of a person and a cup',
                   (7, 54): 'a photo of a person buying a donut', (8, 54): 'a photo of a person carrying a donut',
                   (23, 54): 'a photo of a person eating a donut', (36, 54): 'a photo of a person holding a donut',
                   (54, 54): 'a photo of a person making a donut',
                   (67, 54): 'a photo of a person picking up a donut',
                   (89, 54): 'a photo of a person smelling a donut', (57, 54): 'a photo of a person and a donut',
                   (26, 20): 'a photo of a person feeding an elephant',
                   (36, 20): 'a photo of a person holding an elephant',
                   (38, 20): 'a photo of a person hosing an elephant',
                   (39, 20): 'a photo of a person hugging an elephant',
                   (45, 20): 'a photo of a person kissing an elephant',
                   (37, 20): 'a photo of a person hopping on an elephant',
                   (65, 20): 'a photo of a person petting an elephant',
                   (76, 20): 'a photo of a person riding an elephant',
                   (110, 20): 'a photo of a person walking an elephant',
                   (111, 20): 'a photo of a person washing an elephant',
                   (112, 20): 'a photo of a person watching an elephant',
                   (57, 20): 'a photo of a person and an elephant',
                   (39, 10): 'a photo of a person hugging a fire hydrant',
                   (41, 10): 'a photo of a person inspecting a fire hydrant',
                   (58, 10): 'a photo of a person opening a fire hydrant',
                   (61, 10): 'a photo of a person painting a fire hydrant',
                   (57, 10): 'a photo of a person and a fire hydrant',
                   (36, 42): 'a photo of a person holding a fork', (50, 42): 'a photo of a person lifting a fork',
                   (95, 42): 'a photo of a person sticking a fork', (48, 42): 'a photo of a person licking a fork',
                   (111, 42): 'a photo of a person washing a fork', (57, 42): 'a photo of a person and a fork',
                   (2, 29): 'a photo of a person blocking a frisbee',
                   (9, 29): 'a photo of a person catching a frisbee',
                   (36, 29): 'a photo of a person holding a frisbee',
                   (90, 29): 'a photo of a person spinning a frisbee',
                   (104, 29): 'a photo of a person throwing a frisbee',
                   (57, 29): 'a photo of a person and a frisbee', (26, 23): 'a photo of a person feeding a giraffe',
                   (45, 23): 'a photo of a person kissing a giraffe',
                   (65, 23): 'a photo of a person petting a giraffe',
                   (76, 23): 'a photo of a person riding a giraffe',
                   (112, 23): 'a photo of a person watching a giraffe',
                   (57, 23): 'a photo of a person and a giraffe',
                   (36, 78): 'a photo of a person holding a hair drier',
                   (59, 78): 'a photo of a person operating a hair drier',
                   (75, 78): 'a photo of a person repairing a hair drier',
                   (57, 78): 'a photo of a person and a hair drier',
                   (8, 26): 'a photo of a person carrying a handbag',
                   (36, 26): 'a photo of a person holding a handbag',
                   (41, 26): 'a photo of a person inspecting a handbag',
                   (57, 26): 'a photo of a person and a handbag', (8, 52): 'a photo of a person carrying a hot dog',
                   (14, 52): 'a photo of a person cooking a hot dog',
                   (15, 52): 'a photo of a person cutting a hot dog',
                   (23, 52): 'a photo of a person eating a hot dog',
                   (36, 52): 'a photo of a person holding a hot dog',
                   (54, 52): 'a photo of a person making a hot dog', (57, 52): 'a photo of a person and a hot dog',
                   (8, 66): 'a photo of a person carrying a keyboard',
                   (12, 66): 'a photo of a person cleaning a keyboard',
                   (36, 66): 'a photo of a person holding a keyboard',
                   (109, 66): 'a photo of a person typing on a keyboard',
                   (57, 66): 'a photo of a person and a keyboard', (1, 33): 'a photo of a person assembling a kite',
                   (8, 33): 'a photo of a person carrying a kite', (30, 33): 'a photo of a person flying a kite',
                   (36, 33): 'a photo of a person holding a kite',
                   (41, 33): 'a photo of a person inspecting a kite',
                   (47, 33): 'a photo of a person launching a kite', (70, 33): 'a photo of a person pulling a kite',
                   (57, 33): 'a photo of a person and a kite', (16, 43): 'a photo of a person cutting with a knife',
                   (36, 43): 'a photo of a person holding a knife',
                   (95, 43): 'a photo of a person sticking a knife',
                   (111, 43): 'a photo of a person washing a knife',
                   (115, 43): 'a photo of a person wielding a knife',
                   (48, 43): 'a photo of a person licking a knife', (57, 43): 'a photo of a person and a knife',
                   (36, 63): 'a photo of a person holding a laptop',
                   (58, 63): 'a photo of a person opening a laptop',
                   (73, 63): 'a photo of a person reading a laptop',
                   (75, 63): 'a photo of a person repairing a laptop',
                   (109, 63): 'a photo of a person typing on a laptop',
                   (57, 63): 'a photo of a person and a laptop',
                   (12, 68): 'a photo of a person cleaning a microwave',
                   (58, 68): 'a photo of a person opening a microwave',
                   (59, 68): 'a photo of a person operating a microwave',
                   (57, 68): 'a photo of a person and a microwave',
                   (13, 64): 'a photo of a person controlling a mouse',
                   (36, 64): 'a photo of a person holding a mouse',
                   (75, 64): 'a photo of a person repairing a mouse', (57, 64): 'a photo of a person and a mouse',
                   (7, 49): 'a photo of a person buying an orange',
                   (15, 49): 'a photo of a person cutting an orange',
                   (23, 49): 'a photo of a person eating an orange',
                   (36, 49): 'a photo of a person holding an orange',
                   (41, 49): 'a photo of a person inspecting an orange',
                   (64, 49): 'a photo of a person peeling an orange',
                   (66, 49): 'a photo of a person picking an orange',
                   (91, 49): 'a photo of a person squeezing an orange',
                   (111, 49): 'a photo of a person washing an orange',
                   (57, 49): 'a photo of a person and an orange', (12, 69): 'a photo of a person cleaning an oven',
                   (36, 69): 'a photo of a person holding an oven',
                   (41, 69): 'a photo of a person inspecting an oven',
                   (58, 69): 'a photo of a person opening an oven',
                   (75, 69): 'a photo of a person repairing an oven',
                   (59, 69): 'a photo of a person operating an oven', (57, 69): 'a photo of a person and an oven',
                   (11, 12): 'a photo of a person checking a parking meter',
                   (63, 12): 'a photo of a person paying a parking meter',
                   (75, 12): 'a photo of a person repairing a parking meter',
                   (57, 12): 'a photo of a person and a parking meter',
                   (7, 53): 'a photo of a person buying a pizza', (8, 53): 'a photo of a person carrying a pizza',
                   (14, 53): 'a photo of a person cooking a pizza', (15, 53): 'a photo of a person cutting a pizza',
                   (23, 53): 'a photo of a person eating a pizza', (36, 53): 'a photo of a person holding a pizza',
                   (54, 53): 'a photo of a person making a pizza',
                   (67, 53): 'a photo of a person picking up a pizza',
                   (88, 53): 'a photo of a person sliding a pizza',
                   (89, 53): 'a photo of a person smelling a pizza', (57, 53): 'a photo of a person and a pizza',
                   (12, 72): 'a photo of a person cleaning a refrigerator',
                   (36, 72): 'a photo of a person holding a refrigerator',
                   (56, 72): 'a photo of a person moving a refrigerator',
                   (58, 72): 'a photo of a person opening a refrigerator',
                   (57, 72): 'a photo of a person and a refrigerator',
                   (36, 65): 'a photo of a person holding a remote',
                   (68, 65): 'a photo of a person pointing a remote',
                   (99, 65): 'a photo of a person swinging a remote', (57, 65): 'a photo of a person and a remote',
                   (8, 48): 'a photo of a person carrying a sandwich',
                   (14, 48): 'a photo of a person cooking a sandwich',
                   (15, 48): 'a photo of a person cutting a sandwich',
                   (23, 48): 'a photo of a person eating a sandwich',
                   (36, 48): 'a photo of a person holding a sandwich',
                   (54, 48): 'a photo of a person making a sandwich',
                   (57, 48): 'a photo of a person and a sandwich',
                   (16, 76): 'a photo of a person cutting with a scissors',
                   (36, 76): 'a photo of a person holding a scissors',
                   (58, 76): 'a photo of a person opening a scissors',
                   (57, 76): 'a photo of a person and a scissors', (12, 71): 'a photo of a person cleaning a sink',
                   (75, 71): 'a photo of a person repairing a sink',
                   (111, 71): 'a photo of a person washing a sink', (57, 71): 'a photo of a person and a sink',
                   (8, 36): 'a photo of a person carrying a skateboard',
                   (28, 36): 'a photo of a person flipping a skateboard',
                   (32, 36): 'a photo of a person grinding a skateboard',
                   (36, 36): 'a photo of a person holding a skateboard',
                   (43, 36): 'a photo of a person jumping a skateboard',
                   (67, 36): 'a photo of a person picking up a skateboard',
                   (76, 36): 'a photo of a person riding a skateboard',
                   (87, 36): 'a photo of a person sitting on a skateboard',
                   (93, 36): 'a photo of a person standing on a skateboard',
                   (57, 36): 'a photo of a person and a skateboard',
                   (0, 30): 'a photo of a person adjusting a skis', (8, 30): 'a photo of a person carrying a skis',
                   (36, 30): 'a photo of a person holding a skis',
                   (41, 30): 'a photo of a person inspecting a skis',
                   (43, 30): 'a photo of a person jumping a skis',
                   (67, 30): 'a photo of a person picking up a skis',
                   (75, 30): 'a photo of a person repairing a skis', (76, 30): 'a photo of a person riding a skis',
                   (93, 30): 'a photo of a person standing on a skis',
                   (114, 30): 'a photo of a person wearing a skis', (57, 30): 'a photo of a person and a skis',
                   (0, 31): 'a photo of a person adjusting a snowboard',
                   (8, 31): 'a photo of a person carrying a snowboard',
                   (32, 31): 'a photo of a person grinding a snowboard',
                   (36, 31): 'a photo of a person holding a snowboard',
                   (43, 31): 'a photo of a person jumping a snowboard',
                   (76, 31): 'a photo of a person riding a snowboard',
                   (93, 31): 'a photo of a person standing on a snowboard',
                   (114, 31): 'a photo of a person wearing a snowboard',
                   (57, 31): 'a photo of a person and a snowboard', (36, 44): 'a photo of a person holding a spoon',
                   (48, 44): 'a photo of a person licking a spoon',
                   (111, 44): 'a photo of a person washing a spoon',
                   (85, 44): 'a photo of a person sipping a spoon', (57, 44): 'a photo of a person and a spoon',
                   (2, 32): 'a photo of a person blocking a sports ball',
                   (8, 32): 'a photo of a person carrying a sports ball',
                   (9, 32): 'a photo of a person catching a sports ball',
                   (19, 32): 'a photo of a person dribbling a sports ball',
                   (35, 32): 'a photo of a person hitting a sports ball',
                   (36, 32): 'a photo of a person holding a sports ball',
                   (41, 32): 'a photo of a person inspecting a sports ball',
                   (44, 32): 'a photo of a person kicking a sports ball',
                   (67, 32): 'a photo of a person picking up a sports ball',
                   (81, 32): 'a photo of a person serving a sports ball',
                   (84, 32): 'a photo of a person signing a sports ball',
                   (90, 32): 'a photo of a person spinning a sports ball',
                   (104, 32): 'a photo of a person throwing a sports ball',
                   (57, 32): 'a photo of a person and a sports ball',
                   (36, 11): 'a photo of a person holding a stop sign',
                   (94, 11): 'a photo of a person standing under a stop sign',
                   (97, 11): 'a photo of a person stopping at a stop sign',
                   (57, 11): 'a photo of a person and a stop sign',
                   (8, 28): 'a photo of a person carrying a suitcase',
                   (18, 28): 'a photo of a person dragging a suitcase',
                   (36, 28): 'a photo of a person holding a suitcase',
                   (39, 28): 'a photo of a person hugging a suitcase',
                   (52, 28): 'a photo of a person loading a suitcase',
                   (58, 28): 'a photo of a person opening a suitcase',
                   (60, 28): 'a photo of a person packing a suitcase',
                   (67, 28): 'a photo of a person picking up a suitcase',
                   (116, 28): 'a photo of a person zipping a suitcase',
                   (57, 28): 'a photo of a person and a suitcase',
                   (8, 37): 'a photo of a person carrying a surfboard',
                   (18, 37): 'a photo of a person dragging a surfboard',
                   (36, 37): 'a photo of a person holding a surfboard',
                   (41, 37): 'a photo of a person inspecting a surfboard',
                   (43, 37): 'a photo of a person jumping a surfboard',
                   (49, 37): 'a photo of a person lying on a surfboard',
                   (52, 37): 'a photo of a person loading a surfboard',
                   (76, 37): 'a photo of a person riding a surfboard',
                   (93, 37): 'a photo of a person standing on a surfboard',
                   (87, 37): 'a photo of a person sitting on a surfboard',
                   (111, 37): 'a photo of a person washing a surfboard',
                   (57, 37): 'a photo of a person and a surfboard',
                   (8, 77): 'a photo of a person carrying a teddy bear',
                   (36, 77): 'a photo of a person holding a teddy bear',
                   (39, 77): 'a photo of a person hugging a teddy bear',
                   (45, 77): 'a photo of a person kissing a teddy bear',
                   (57, 77): 'a photo of a person and a teddy bear',
                   (8, 38): 'a photo of a person carrying a tennis racket',
                   (36, 38): 'a photo of a person holding a tennis racket',
                   (41, 38): 'a photo of a person inspecting a tennis racket',
                   (99, 38): 'a photo of a person swinging a tennis racket',
                   (57, 38): 'a photo of a person and a tennis racket',
                   (0, 27): 'a photo of a person adjusting a tie', (15, 27): 'a photo of a person cutting a tie',
                   (36, 27): 'a photo of a person holding a tie', (41, 27): 'a photo of a person inspecting a tie',
                   (70, 27): 'a photo of a person pulling a tie', (105, 27): 'a photo of a person tying a tie',
                   (114, 27): 'a photo of a person wearing a tie', (57, 27): 'a photo of a person and a tie',
                   (36, 70): 'a photo of a person holding a toaster',
                   (59, 70): 'a photo of a person operating a toaster',
                   (75, 70): 'a photo of a person repairing a toaster',
                   (57, 70): 'a photo of a person and a toaster', (12, 61): 'a photo of a person cleaning a toilet',
                   (29, 61): 'a photo of a person flushing a toilet',
                   (58, 61): 'a photo of a person opening a toilet',
                   (75, 61): 'a photo of a person repairing a toilet',
                   (87, 61): 'a photo of a person sitting on a toilet',
                   (93, 61): 'a photo of a person standing on a toilet',
                   (111, 61): 'a photo of a person washing a toilet', (57, 61): 'a photo of a person and a toilet',
                   (6, 79): 'a photo of a person brushing with a toothbrush',
                   (36, 79): 'a photo of a person holding a toothbrush',
                   (111, 79): 'a photo of a person washing a toothbrush',
                   (57, 79): 'a photo of a person and a toothbrush',
                   (42, 9): 'a photo of a person installing a traffic light',
                   (75, 9): 'a photo of a person repairing a traffic light',
                   (94, 9): 'a photo of a person standing under a traffic light',
                   (97, 9): 'a photo of a person stopping at a traffic light',
                   (57, 9): 'a photo of a person and a traffic light',
                   (17, 7): 'a photo of a person directing a truck', (21, 7): 'a photo of a person driving a truck',
                   (41, 7): 'a photo of a person inspecting a truck',
                   (52, 7): 'a photo of a person loading a truck', (75, 7): 'a photo of a person repairing a truck',
                   (76, 7): 'a photo of a person riding a truck', (87, 7): 'a photo of a person sitting on a truck',
                   (111, 7): 'a photo of a person washing a truck', (57, 7): 'a photo of a person and a truck',
                   (8, 25): 'a photo of a person carrying a umbrella',
                   (36, 25): 'a photo of a person holding a umbrella',
                   (53, 25): 'a photo of a person losing a umbrella',
                   (58, 25): 'a photo of a person opening a umbrella',
                   (75, 25): 'a photo of a person repairing a umbrella',
                   (82, 25): 'a photo of a person setting a umbrella',
                   (94, 25): 'a photo of a person standing under a umbrella',
                   (57, 25): 'a photo of a person and a umbrella', (36, 75): 'a photo of a person holding a vase',
                   (54, 75): 'a photo of a person making a vase', (61, 75): 'a photo of a person painting a vase',
                   (57, 75): 'a photo of a person and a vase', (27, 40): 'a photo of a person filling a wine glass',
                   (36, 40): 'a photo of a person holding a wine glass',
                   (85, 40): 'a photo of a person sipping a wine glass',
                   (106, 40): 'a photo of a person toasting a wine glass',
                   (48, 40): 'a photo of a person licking a wine glass',
                   (111, 40): 'a photo of a person washing a wine glass',
                   (57, 40): 'a photo of a person and a wine glass',
                   (26, 22): 'a photo of a person feeding a zebra', (36, 22): 'a photo of a person holding a zebra',
                   (65, 22): 'a photo of a person petting a zebra',
                   (112, 22): 'a photo of a person watching a zebra', (57, 22): 'a photo of a person and a zebra'}

hico_obj_text_label = [(0, 'a photo of a person'), (1, 'a photo of a bicycle'), (2, 'a photo of a car'),
                       (3, 'a photo of a motorcycle'), (4, 'a photo of an airplane'), (5, 'a photo of a bus'),
                       (6, 'a photo of a train'), (7, 'a photo of a truck'), (8, 'a photo of a boat'),
                       (9, 'a photo of a traffic light'), (10, 'a photo of a fire hydrant'),
                       (11, 'a photo of a stop sign'), (12, 'a photo of a parking meter'), (13, 'a photo of a bench'),
                       (14, 'a photo of a bird'), (15, 'a photo of a cat'), (16, 'a photo of a dog'),
                       (17, 'a photo of a horse'), (18, 'a photo of a sheep'), (19, 'a photo of a cow'),
                       (20, 'a photo of an elephant'), (21, 'a photo of a bear'), (22, 'a photo of a zebra'),
                       (23, 'a photo of a giraffe'), (24, 'a photo of a backpack'), (25, 'a photo of a umbrella'),
                       (26, 'a photo of a handbag'), (27, 'a photo of a tie'), (28, 'a photo of a suitcase'),
                       (29, 'a photo of a frisbee'), (30, 'a photo of a skis'), (31, 'a photo of a snowboard'),
                       (32, 'a photo of a sports ball'), (33, 'a photo of a kite'), (34, 'a photo of a baseball bat'),
                       (35, 'a photo of a baseball glove'), (36, 'a photo of a skateboard'),
                       (37, 'a photo of a surfboard'), (38, 'a photo of a tennis racket'), (39, 'a photo of a bottle'),
                       (40, 'a photo of a wine glass'), (41, 'a photo of a cup'), (42, 'a photo of a fork'),
                       (43, 'a photo of a knife'), (44, 'a photo of a spoon'), (45, 'a photo of a bowl'),
                       (46, 'a photo of a banana'), (47, 'a photo of an apple'), (48, 'a photo of a sandwich'),
                       (49, 'a photo of an orange'), (50, 'a photo of a broccoli'), (51, 'a photo of a carrot'),
                       (52, 'a photo of a hot dog'), (53, 'a photo of a pizza'), (54, 'a photo of a donut'),
                       (55, 'a photo of a cake'), (56, 'a photo of a chair'), (57, 'a photo of a couch'),
                       (58, 'a photo of a potted plant'), (59, 'a photo of a bed'), (60, 'a photo of a dining table'),
                       (61, 'a photo of a toilet'), (62, 'a photo of a tv'), (63, 'a photo of a laptop'),
                       (64, 'a photo of a mouse'), (65, 'a photo of a remote'), (66, 'a photo of a keyboard'),
                       (67, 'a photo of a cell phone'), (68, 'a photo of a microwave'), (69, 'a photo of an oven'),
                       (70, 'a photo of a toaster'), (71, 'a photo of a sink'), (72, 'a photo of a refrigerator'),
                       (73, 'a photo of a book'), (74, 'a photo of a clock'), (75, 'a photo of a vase'),
                       (76, 'a photo of a scissors'), (77, 'a photo of a teddy bear'), (78, 'a photo of a hair drier'),
                       (79, 'a photo of a toothbrush'), (80, 'a photo of nothing')]

def build_test_dataset(test_anno, image_folder):
    with open(test_anno, 'r') as f:
        annotations_eval = json.load(f)

    coco_class_dict = {
        1: 'person',2: 'bicycle',3: 'car',4: 'motorcycle',5: 'airplane',6: 'bus',7: 'train',8: 'truck',9: 'boat',10: 'traffic light',
        11: 'fire hydrant',13: 'stop sign',14: 'parking meter',15: 'bench',16: 'bird',17: 'cat',18: 'dog',19: 'horse',20: 'sheep',
        21: 'cow',22: 'elephant',23: 'bear',24: 'zebra',25: 'giraffe',27: 'backpack',28: 'umbrella',31: 'handbag',32: 'tie',33: 'suitcase',
        34: 'frisbee',35: 'skis',36: 'snowboard',37: 'sports ball',38: 'kite',39: 'baseball bat',40: 'baseball glove',41: 'skateboard',42: 'surfboard',
        43: 'tennis racket',44: 'bottle',46: 'wine glass',47: 'cup',48: 'fork',49: 'knife',50: 'spoon',51: 'bowl',52: 'banana',53: 'apple',
        54: 'sandwich',55: 'orange',56: 'broccoli',57: 'carrot',58: 'hot dog',59: 'pizza',60: 'donut',61: 'cake',62: 'chair',63: 'couch',
        64: 'potted plant',65: 'bed',67: 'dining table',70: 'toilet',72: 'tv',73: 'laptop',74: 'mouse',75: 'remote',76: 'keyboard',77: 'cell phone',
        78: 'microwave',79: 'oven',80: 'toaster',81: 'sink',82: 'refrigerator',84: 'book',85: 'clock',86: 'vase',87: 'scissors',88: 'teddy bear',
        89: 'hair drier',90: 'toothbrush'
    }
    coco_class_dict = list(coco_class_dict.values())

    hico_verb_dict = {
        1: 'adjust', 2: 'assemble', 3: 'block',4: 'blow',5: 'board',6: 'break',7: 'brush with',8: 'buy',9: 'carry',
        10: 'catch',11: 'chase',12: 'check',13: 'clean',14: 'control',15: 'cook',16: 'cut',17: 'cut with',18: 'direct',19: 'drag',
        20: 'dribble',21: 'drink with',22: 'drive',23: 'dry',24: 'eat',25: 'eat at',26: 'exit',27: 'feed',28: 'fill',29: 'flip',
        30: 'flush',31: 'fly',32: 'greet',33: 'grind',34: 'groom',35: 'herd',36: 'hit',37: 'hold',38: 'hop on',39: 'hose',
        40: 'hug',41: 'hunt',42: 'inspect',43: 'install',44: 'jump',45: 'kick',46: 'kiss',47: 'lasso',48: 'launch',49: 'lick',
        50: 'lie on',51: 'lift',52: 'light',53: 'load',54: 'lose',55: 'make',56: 'milk',57: 'move',58: 'no interaction',59: 'open',
        60: 'operate',61: 'pack',62: 'paint',63: 'park',64: 'pay',65: 'peel',66: 'pet',67: 'pick',68: 'pick up',69: 'point',
        70: 'pour',71: 'pull',72: 'push',73: 'race',74: 'read',75: 'release',76: 'repair',77: 'ride',78: 'row',79: 'run',
        80: 'sail',81: 'scratch',82: 'serve',83: 'set',84: 'shear',85: 'sign',86: 'sip',87: 'sit at',88: 'sit on',89: 'slide',
        90: 'smell',91: 'spin',92: 'squeeze',93: 'stab',94: 'stand on',95: 'stand under',96: 'stick',97: 'stir',98: 'stop at',99: 'straddle',
        100: 'swing',101: 'tag',102: 'talk on',103: 'teach',104: 'text on',105: 'throw',106: 'tie',107: 'toast',108: 'train',109: 'turn',
        110: 'type on',111: 'walk',112: 'wash',113: 'watch',114: 'wave',115: 'wear',116: 'wield',117: 'zip'
    }
    hico_verb_dict = list(hico_verb_dict.values())

    _valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                82, 84, 85, 86, 87, 88, 89, 90)
    _valid_verb_ids = list(range(1, 118))
    text_label_ids = list(hico_text_label.keys())

    all_data_eval = []
    image_folder_eval = image_folder
    for img_anno in annotations_eval:
        boxes = [obj['bbox'] for obj in img_anno['annotations']]
        classes = [(i, _valid_obj_ids.index(obj['category_id'])) for i, obj in enumerate(img_anno['annotations'])]

        kept_box_indices = [label[0] for label in classes]
        classes = [label[1] for label in classes]

        obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], []
        sub_obj_pairs = []
        hoi_labels = []
        for hoi in img_anno['hoi_annotation']:
            if hoi['subject_id'] not in kept_box_indices or hoi['object_id'] not in kept_box_indices:
                continue

            verb_obj_pair = (_valid_verb_ids.index(hoi['category_id']), classes[kept_box_indices.index(hoi['object_id'])])
            if verb_obj_pair not in text_label_ids:
                continue
            verb_obj_pair_name = hico_text_label[verb_obj_pair]

            sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
            if sub_obj_pair in sub_obj_pairs:
                verb_labels[sub_obj_pairs.index(sub_obj_pair)][_valid_verb_ids.index(hoi['category_id'])] = 1
                hoi_labels[sub_obj_pairs.index(sub_obj_pair)][text_label_ids.index(verb_obj_pair)] = 1
            else:
                sub_obj_pairs.append(sub_obj_pair)
                obj_labels.append(classes[kept_box_indices.index(hoi['object_id'])])
                verb_label = [0 for _ in range(len(_valid_verb_ids))]
                hoi_label = [0] * len(text_label_ids)
                hoi_label[text_label_ids.index(verb_obj_pair)] = 1
                verb_label[_valid_verb_ids.index(hoi['category_id'])] = 1
                sub_box = boxes[kept_box_indices.index(hoi['subject_id'])]
                obj_box = boxes[kept_box_indices.index(hoi['object_id'])]

                verb_labels.append(verb_label)
                hoi_labels.append(hoi_label)
                sub_boxes.append(sub_box)
                obj_boxes.append(obj_box)

        # Create solution
        # solution format
        # [{
        #     "human": "[x1, x2, y1, y2]",
        #     "object": "[x1, x2, y1, y2]",
        #     "object class": OBJECT_CLASS,
        #     "verb class": [VERB_CLASS_1, VERB_CLASS_2, ...]
        # }, ...]
        solution = []
        for so_idx, so in enumerate(sub_obj_pairs):
            solution_i = {
                "human": sub_boxes[so_idx],
                "object": obj_boxes[so_idx],
                "object class": "",
                "verb class": []
            }

            hoi_label_arr = np.array(hoi_labels[so_idx])
            for hoi_idx in np.where(hoi_label_arr == 1)[0]:
                verb_obj_pair = text_label_ids[hoi_idx]
                if solution_i["object class"] == "":
                    solution_i["object class"] = coco_class_dict[obj_labels[so_idx]]
                solution_i["verb class"].append(hico_verb_dict[verb_obj_pair[0]])

            solution.append(solution_i)
        if len(solution) == 0:
            continue  # Skip if no valid human-object pairs found

        # change solution to str
        solution = str(json.dumps(solution))

        item = {
            'image_path': [os.path.join(image_folder_eval, img_anno['file_name'])],
            'problem': 'Please provide all bounding boxes of human-object pairs interacting in the image with the object and verb classes.',
            'solution': solution,
        }
        all_data_eval.append(item)
    return all_data_eval


def run_task(task_list, task_id, output_folder, num_gpus=4, model_path="Qwen/Qwen3-VL-4B-Instruct"):
    gpu_id = task_id % num_gpus

    # Load model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=f"cuda:{gpu_id}",
    )
    processor = AutoProcessor.from_pretrained(model_path)


    QUESTION_TEMPLATE_v2_1 = """You are an HOI detection model. Follow this exact structure:

<VALID OBJECT CLASSES>:
<object class>:person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush.

<VALID INTERACTIONS> (object: allowed_verbs):
<object class>: <verb class>, <verb class>, ...
airplane: board, direct, exit, fly, inspect, load, ride, sit on, wash, no interaction
bicycle: carry, hold, inspect, jump, hop on, park, push, repair, ride, sit on, straddle, walk, wash, no interaction
bird: chase, feed, hold, pet, release, watch, no interaction
boat: board, drive, exit, inspect, jump, launch, repair, ride, row, sail, sit on, stand on, tie, wash, no interaction
bottle: carry, drink with, hold, inspect, lick, open, pour, no interaction
bus: board, direct, drive, exit, inspect, load, ride, sit on, wash, wave, no interaction
car: board, direct, drive, hose, inspect, jump, load, park, ride, wash, no interaction
cat: dry, feed, hold, hug, kiss, pet, scratch, wash, chase, no interaction
chair: carry, hold, lie on, sit on, stand on, no interaction
couch: carry, lie on, sit on, no interaction
cow: feed, herd, hold, hug, kiss, lasso, milk, pet, ride, walk, no interaction
dining table: clean, eat at, sit at, no interaction
dog: carry, dry, feed, groom, hold, hose, hug, inspect, kiss, pet, run, scratch, straddle, train, walk, wash, chase, no interaction
horse: feed, groom, hold, hug, jump, kiss, load, hop on, pet, race, ride, run, straddle, train, walk, wash, no interaction
motorcycle: hold, inspect, jump, hop on, park, push, race, ride, sit on, straddle, turn, walk, wash, no interaction
person: carry, greet, hold, hug, kiss, stab, tag, teach, lick, no interaction
potted plant: carry, hold, hose, no interaction
sheep: carry, feed, herd, hold, hug, kiss, pet, ride, shear, walk, wash, no interaction
train: board, drive, exit, load, ride, sit on, wash, no interaction
tv: control, repair, watch, no interaction
apple: buy, cut, eat, hold, inspect, peel, pick, smell, wash, no interaction
backpack: carry, hold, inspect, open, wear, no interaction
banana: buy, carry, cut, eat, hold, inspect, peel, pick, smell, no interaction
baseball bat: break, carry, hold, sign, swing, throw, wield, no interaction
baseball glove: hold, wear, no interaction
bear: feed, hunt, watch, no interaction
bed: clean, lie on, sit on, no interaction
bench: inspect, lie on, sit on, no interaction
book: carry, hold, open, read, no interaction
bowl: hold, stir, wash, lick, no interaction
broccoli: cut, eat, hold, smell, stir, wash, no interaction
cake: blow, carry, cut, eat, hold, light, make, pick up, no interaction
carrot: carry, cook, cut, eat, hold, peel, smell, stir, wash, no interaction
cell phone: carry, hold, read, repair, talk on, text on, no interaction
clock: check, hold, repair, set, no interaction
cup: carry, drink with, hold, inspect, pour, sip, smell, fill, wash, no interaction
donut: buy, carry, eat, hold, make, pick up, smell, no interaction
elephant: feed, hold, hose, hug, kiss, hop on, pet, ride, walk, wash, watch, no interaction
fire hydrant: hug, inspect, open, paint, no interaction
fork: hold, lift, stick, lick, wash, no interaction
frisbee: block, catch, hold, spin, throw, no interaction
giraffe: feed, kiss, pet, ride, watch, no interaction
hair drier: hold, operate, repair, no interaction
handbag: carry, hold, inspect, no interaction
hot dog: carry, cook, cut, eat, hold, make, no interaction
keyboard: carry, clean, hold, type on, no interaction
kite: assemble, carry, fly, hold, inspect, launch, pull, no interaction
knife: cut with, hold, stick, wash, wield, lick, no interaction
laptop: hold, open, read, repair, type on, no interaction
microwave: clean, open, operate, no interaction
mouse: control, hold, repair, no interaction
orange: buy, cut, eat, hold, inspect, peel, pick, squeeze, wash, no interaction
oven: clean, hold, inspect, open, repair, operate, no interaction
parking meter: check, pay, repair, no interaction
pizza: buy, carry, cook, cut, eat, hold, make, pick up, slide, smell, no interaction
refrigerator: clean, hold, move, open, no interaction
remote: hold, point, swing, no interaction
sandwich: carry, cook, cut, eat, hold, make, no interaction
scissors: cut with, hold, open, no interaction
sink: clean, repair, wash, no interaction
skateboard: carry, flip, grind, hold, jump, pick up, ride, sit on, stand on, no interaction
skis: adjust, carry, hold, inspect, jump, pick up, repair, ride, stand on, wear, no interaction
snowboard: adjust, carry, grind, hold, jump, ride, stand on, wear, no interaction
spoon: hold, lick, wash, sip, no interaction
sports ball: block, carry, catch, dribble, hit, hold, inspect, kick, pick up, serve, sign, spin, throw, no interaction
stop sign: hold, stand under, stop at, no interaction
suitcase: carry, drag, hold, hug, load, open, pack, pick up, zip, no interaction
surfboard: carry, drag, hold, inspect, jump, lie on, load, ride, stand on, sit on, wash, no interaction
teddy bear: carry, hold, hug, kiss, no interaction
tennis racket: carry, hold, inspect, swing, no interaction
tie: adjust, cut, hold, inspect, pull, tie, wear, no interaction
toaster: hold, operate, repair, no interaction
toilet: clean, flush, open, repair, sit on, stand on, wash, no interaction
toothbrush: brush with, hold, wash, no interaction
traffic light: install, repair, stand under, stop at, no interaction
truck: direct, drive, inspect, load, repair, ride, sit on, wash, no interaction
umbrella: carry, hold, lose, open, repair, set, stand under, no interaction
vase: hold, make, paint, no interaction
wine glass: fill, hold, sip, toast, lick, wash, no interaction
zebra: feed, hold, pet, watch, no interaction

Thinking Process:
1. How many people are in the image?
2. What are those people doing?
3. What <object class> in <VALID OBJECT CLASSES> are each people interacting with?
4. Find existing <verb class> combinations from <VALID INTERACTIONS> for each <object class>.

EXAMPLE OUTPUT FORMAT:
<think>
reasoning process here
</think>
<answer>
[
    {
        "human":  [x1, y1, x2, y2],
        "object": [x1, y1, x2, y2],
        "object class": "<object class>",
        "verb class": "<verb class>, <verb class>, ... "
    },
    ...
]
</answer>"""

    preds = []
    gt_paths = []
    outputs = []
    save_internal = 25
    for val_idx, val_i in tqdm(enumerate(task_list), total=len(task_list), desc=f"Task {task_id}: "):
        # Image path
        image_path = val_i['image_path'][0]

        # Build messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"file://{image_path}"}
                    },
                    {
                        "type": "text",
                        "text": QUESTION_TEMPLATE_v2_1
                    }
                ]
            }
        ]

        # Apply chat template
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process inputs
        img = Image.open(image_path)
        inputs = processor(
            text=text,
            images=[img],
            return_tensors="pt",
        ).to(f"cuda:{gpu_id}")

        # Generate output
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
        )

        # Decode output
        output_text = processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        outputs.append(output_text)

        # Process output
        hoi_data = extract_and_validate_hoi_output(output_text)
        base_name = os.path.splitext(image_path)[0].split('/')[-1]

        preds.append(hoi_data)
        gt_paths.append({"file_name": base_name})

        if val_idx % save_internal == 0 and val_idx > 0:
            pred_data = {
                "outputs": outputs,
                "preds": preds,
                "gt_paths": gt_paths
            }

            date_now = datetime.now().strftime("%d-%H-%M-%S-%f")

            with open(f"{output_folder}/pred_{task_id}_{date_now}.json", "w") as f:
                json.dump(pred_data, f, indent=4)

            preds = []
            gt_paths = []
            outputs = []
        elif val_idx == len(task_list) - 1:
            pred_data = {
                "outputs": outputs,
                "preds": preds,
                "gt_paths": gt_paths
            }

            date_now = datetime.now().strftime("%d-%H-%M-%S-%f")

            with open(f"{output_folder}/pred_{task_id}_{date_now}.json", "w") as f:
                json.dump(pred_data, f, indent=4)


if __name__ == "__main__":
    args = parse_args()

    # Validate section parameters
    if args.section < 0 or args.section >= args.max_section:
        raise ValueError(f"Section index {args.section} must be in range [0, {args.max_section-1}]")

    MODEL_PATH = args.model_path
    DEVICE_NUM = args.num_gpus

    output_folder = f"{args.output_folder}"
    if args.max_section > 1:
        output_folder = f"{output_folder}/section{args.section}"
    os.makedirs(output_folder, exist_ok=True)

    existing_merged_json = args.existing_merged_json
    num_tasks = DEVICE_NUM

    mp.set_start_method('spawn', force=True)

    val_dataset = build_test_dataset(args.test_anno, args.image_folder)

    # Remove already processed data first
    if existing_merged_json != "":
        val_dataset_cur = []
        with open(existing_merged_json, 'r') as f:
            data = json.load(f)
            processed_paths = [p["file_name"] for p in data['gt_paths']]

        for val_i in val_dataset:
            image_path = val_i['image_path'][0]
            base_name = os.path.splitext(image_path)[0].split('/')[-1]
            if not base_name in processed_paths:
                val_dataset_cur.append(val_i)
        val_dataset = val_dataset_cur
        print(f"After filtering processed data: {len(val_dataset)} images remaining")

    # Apply section filtering after removing processed data
    if args.max_section > 1:
        total_images = len(val_dataset)
        images_per_section = total_images // args.max_section
        start_idx = args.section * images_per_section

        if args.section == args.max_section - 1:
            # Last section gets remaining images
            end_idx = total_images
        else:
            end_idx = start_idx + images_per_section

        val_dataset = val_dataset[start_idx:end_idx]
        print(f"Processing section {args.section} of {args.max_section}: images {start_idx} to {end_idx-1} ({len(val_dataset)} images)")

    print("Total number of images to process: ", len(val_dataset))

    task_lists = []
    total_num = len(val_dataset)
    per_task_num = total_num // num_tasks
    for i in range(num_tasks):
        if i == num_tasks - 1:
            task_lists.append(val_dataset[i*per_task_num:])
        else:
            task_lists.append(val_dataset[i*per_task_num:(i+1)*per_task_num])

    processes = []
    for i in range(num_tasks):
        p = mp.Process(target=run_task, args=(task_lists[i], i, output_folder, DEVICE_NUM, MODEL_PATH))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
