import argparse
import json
# from multiprocessing import set_start_method    # local
import os
import sys
from random import shuffle
import numpy as np
from typing import Type
import time
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import face_detector
from face_detector import VideoDataset
from face_detector import VideoFaceDetector
from utils import get_video_paths, get_method
import argparse
import torch
import warnings
warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


def process_videos(videos, detector_cls: Type[VideoFaceDetector], selected_dataset, opt):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        detector = face_detector.__dict__[detector_cls](device=device)

        dataset = VideoDataset(videos)

        # loader = DataLoader(dataset, shuffle=False, num_workers=12, batch_size=1, collate_fn=lambda x: x)    # local
        loader = DataLoader(dataset, shuffle=False, num_workers=opt.processes, batch_size=int(opt.batch_size), collate_fn=lambda x: x)
        missed_videos = []
        i = 1
        k = 0
        end = time.time()
        for item in tqdm(loader):
            print('***Item - stats: (len, size)', len(item), sys.getsizeof(item))
        # for item in loader:
            start = time.time()
            print('------loop2loop: ', start - end)
            print('\n ---------------------------------------------------')
            print(f'***item {i} being processing from {len(dataset)}...')
            result = {}
            video, indices, frames = item[0]
            print(f"***path: ", str(video).split('Faceforensic')[1], len(indices))
            if selected_dataset == 1:
                method = get_method(video, opt.data_path)
                if opt.output == "":
                    out_dir = os.path.join(opt.data_path, "boxes", method)
                else:
                    out_dir = os.path.join(opt.output, "boxes", method)
            else:
                out_dir = os.path.join(opt.data_path, "boxes")

            id = os.path.splitext(os.path.basename(video))[0]

            if os.path.exists(out_dir) and "{}.json".format(id) in os.listdir(out_dir):
                continue
            
            t1 = time.time()
            print('------process_videos initialization: ', t1 - start)
            batches = [frames[i:i + detector._batch_size] for i in range(0, len(frames), detector._batch_size)]
            t2 = time.time()
            for j, frames in enumerate(batches):
                result.update({int(j * detector._batch_size) + i : b for i, b in zip(indices, detector._detect_faces(frames))})
            print("***looped over batches. batch length :", len(batches))
            t3 = time.time()
            print('------detection: ', t3 - t2)
            os.makedirs(out_dir, exist_ok=True)
            if len(result) > 0:
                print('***writing results to dir - stats: (len, size)', len(result), sys.getsizeof(result))
                with open(os.path.join(out_dir, "{}.json".format(id)), "w") as f:
                    json.dump(result, f)
                print(f'***writing item {i} done.')
                k += 1
            else:
                missed_videos.append(id)
                print(f'***missed video in item {i}; id: ', id, "result: ", result)
            t4 = time.time()
            print('------saving: ', t4 - t3)
            print(f'***Success: {k} out of {i} items')
            i += 1
            end = time.time()
            print('------sample total: ', end - start)

        if len(missed_videos) > 0:
            print("The detector did not find faces inside the following videos:")
            print(id)
            print("We suggest to re-run the code decreasing the detector threshold.")


    except Exception as e:
        print('***Error Occured: ', e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="DFDC", type=str,
                        help='Dataset (DFDC / FACEFORENSICS)')
    parser.add_argument('--data_path', default='', type=str,
                        help='Videos directory')
    parser.add_argument("--detector-type", help="Type of the detector", default="FacenetDetector",
                        choices=["FacenetDetector"])
    parser.add_argument("--processes", help="Number of processes", default=2)
    parser.add_argument("--output", help="Number of processes", default='')
    parser.add_argument("--batch_size", help="Number of processes", default=1)
    opt = parser.parse_args()
    print(opt)


    if opt.dataset.upper() == "DFDC":
        dataset = 0
    else:
        dataset = 1

    all_paths = []
    videos_paths = []
    if dataset == 1:
        all_paths = get_video_paths(opt.data_path, dataset)

    
        already_extracted = []
        from pathlib import Path

        for path in Path(opt.output).rglob('*.json'):
            already_extracted.append({'name': path.name, 'method': str(path).split('/')[-2]})

        for video_path in all_paths:
            video_name = (video_path.split(".")[0] + ".json").split('/')[-1]
            video_type = video_path.split('/')[-4]

            if {'name': video_name, 'method': video_type} in already_extracted or video_type == 'actors':
                continue

            shuffle(videos_paths)
            videos_paths.append(video_path)

        print(f'{len(already_extracted)} already processed.')
        print(f'{len(videos_paths)} from the total {len(all_paths)} videos ...')

    else:
        os.makedirs(os.path.join(opt.data_path, "boxes"), exist_ok=True)
        already_extracted = os.listdir(os.path.join(opt.data_path, "boxes"))
        for folder in os.listdir(opt.data_path):
            if "boxes" not in folder and "zip" not in folder:
                if os.path.isdir(os.path.join(opt.data_path, folder)): # For training and test set
                    for video_name in os.listdir(os.path.join(opt.data_path, folder)):
                        if video_name.split(".")[0] + ".json" in already_extracted:
                            continue
                        videos_paths.append(os.path.join(opt.data_path, folder, video_name))
                else: # For validation set
                    videos_paths.append(os.path.join(opt.data_path, folder))

    process_videos(videos_paths, opt.detector_type, dataset, opt)


if __name__ == "__main__":
    # set_start_method('fork')    # local
    main()
