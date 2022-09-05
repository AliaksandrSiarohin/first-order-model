import warnings
warnings.filterwarnings("ignore")
import os
import glob
import argparse
import random
from typing import Dict, List
from imageio import mimread, imwrite
import numpy as np
import subprocess
import multiprocessing
from tqdm import tqdm
from multiprocessing.dummy import Pool
from functools import partial

TOTAL_ID = 6112

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/brianw0924/SN850/vox2_mp4/train")
    parser.add_argument("--save_dir", type=str, default="./save")
    parser.add_argument("--num_driving_id", type=int, default=10)
    parser.add_argument("--videos_per_driving_id", type=int, default=10)
    parser.add_argument("--num_source_id", type=int, default=300)
    parser.add_argument("--images_per_source_id", type=int, default=1, help="At most") # lowest priority
    parser.add_argument("--seed", type=int, default=7414)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    assert args.num_driving_id + args.num_source_id <= TOTAL_ID
    if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
            os.mkdir(os.path.join(args.save_dir, "syn"))
            os.mkdir(os.path.join(args.save_dir, "meta"))
    random.seed(args.seed)
    return args


def run(driving_id: str, driving_videos: Dict[str, Dict], source_videos: Dict[str, Dict], idx=0):
    tqdm.write(driving_id)
    if not os.path.exists(os.path.join(args.save_dir, "syn", f"driving_{driving_id}")):
        os.mkdir(os.path.join(args.save_dir, "syn", f"driving_{driving_id}"))

    with open(os.path.join(args.save_dir, "meta", f"{driving_id}_driving_videos.txt"), 'w') as f:
                f.write(f"driving_id,driving_video_id,driving_video_clip,source_id,source_video_id,source_video_clip\n")

    try:
        driving_video_ids = random.sample(list(driving_videos[driving_id].keys()), k=args.videos_per_driving_id)
    except:
        driving_video_ids = list(driving_videos[driving_id].keys())
        
    for driving_video_id in driving_video_ids:
        driving_video_path = random.choice(driving_videos[driving_id][driving_video_id])
        for source_id in tqdm(source_videos.keys()):
            '''
            Sample {args.images_per_source_id} videos
            '''
            try:
                source_video_ids = random.sample(list(source_videos[source_id].keys()), k=args.images_per_source_id)
            except:
                source_video_ids = list(source_videos[source_id].keys())

            for source_video_id in source_video_ids:
                save_path = os.path.join(
                    args.save_dir, "syn", f"driving_{driving_id}",
                    f"{source_video_id}.mp4"
                )
                assert not os.path.exists(save_path), "Duplicate source_video_id. Each source video can only have 1 clip synthesized."
                '''
                Sample 1 clips from video
                '''
                source_video_path = random.choice(source_videos[source_id][source_video_id])
                v = mimread(source_video_path, memtest=False)
                [source_image] = np.array(random.sample(v, k=1))
                source_image_path = os.path.join(args.save_dir, f"tmp_{driving_id}.png")
                imwrite(source_image_path, source_image)
                subprocess.call([
                    "python",
                    "demo.py",
                    "--source_image", source_image_path,
                    "--driving_video", driving_video_path,
                    "--result_video", save_path
                    ])
                idx+=1

                '''
                Write meta info
                '''
                with open(os.path.join(args.save_dir, "meta", f"{driving_id}_driving_videos.txt"), 'a') as f:
                    f.write(f"{driving_id},{driving_video_id},{os.path.basename(driving_video_path)},{source_id},{source_video_id},{os.path.basename(source_video_path)}\n")



def gen_data(args):
        ids = os.listdir(args.data_dir)
        selected_ids = random.sample(ids, k=(args.num_driving_id + args.num_source_id))
        driving_ids, source_ids = selected_ids[:args.num_driving_id], selected_ids[args.num_driving_id:]
        
        driving_videos = {}
        for driving_id in driving_ids:
            driving_videos[driving_id] = {}
            for video_id in os.listdir(os.path.join(args.data_dir, driving_id)):
                if len(os.listdir(os.path.join(args.data_dir, driving_id, video_id))) > 0:
                    driving_videos[driving_id][video_id] = glob.glob(os.path.join(args.data_dir, driving_id, video_id, "*"))

        source_videos = {}
        for source_id in source_ids:
            source_videos[source_id] = {}
            for video_id in os.listdir(os.path.join(args.data_dir, source_id)):
                if len(os.listdir(os.path.join(args.data_dir, source_id, video_id))) > 0:
                    source_videos[source_id][video_id] = glob.glob(os.path.join(args.data_dir, source_id, video_id, "*"))

        with Pool(args.threads) as pool:
            pool.map(
                partial(run, driving_videos=driving_videos, source_videos=source_videos),
                driving_videos.keys()) # Calls run(i) for each element i in range(5)
            pool.close()
            pool.join()

if __name__ == "__main__":
        multiprocessing.set_start_method('spawn', force=True)
        args = get_args()
        gen_data(args)