# TaiChi dataset

The scripst for loading the TaiChi dataset. 

We provide only the id of the corresponding video and the bounding box. Following script will download videos from youtube and crop them according to the provided bounding boxes.

1) Load youtube-dl:
```
wget https://yt-dl.org/downloads/latest/youtube-dl -O youtube-dl
chmod a+rx youtube-dl
```

2) Run script to download videos, there are 2 formats that can be used for storing videos one is .mp4 and another is folder with .png images. While .png images occupy significantly more space, the format is loss-less and have better i/o performance when training.

```
python load_videos.py --metadata taichi-metadata.csv --format .mp4 --out_folder taichi --workers 8
```
select number of workers based on number of cpu avaliable. Note .png format take aproximatly 80GB.
