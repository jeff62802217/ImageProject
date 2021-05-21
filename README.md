# Project for Image and Video Processing

## Summary

This is the PyTorch code for the IVP course project, which implements Image and Video Processing.

This code includes training and testing on UCF-1011.

## Requirements

* [PyTorch](http://pytorch.org/)

```bash
conda install pytorch torchvision cuda80 -c soumith
```

* FFmpeg, FFprobe

```bash
wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
tar xvf ffmpeg-release-64bit-static.tar.xz
cd ./ffmpeg-3.3.3-64bit-static/; sudo cp ffmpeg ffprobe /usr/local/bin;
```

* Python 3



## Preparation

### UCF-101

* Download videos  [here](http://crcv.ucf.edu/data/UCF101.php).
* Convert from avi to jpg files using ```utils/video_jpg_ucf101_hmdb51.py```

```bash
python utils/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
```

* Generate n_frames files using ```utils/n_frames_ucf101_hmdb51.py```

```bash
python utils/n_frames_ucf101_hmdb51.py jpg_video_directory
```



## Running the code

Assume the structure of data directories is the following:

```misc
~/
  data/
    ucf_videos/
      jpg/
        .../ (directories of class names)
          .../ (directories of video names)
            ... (jpg files)
    results/
      save_100.pth
    kinetics.json
```

Confirm all options.

```bash
python main.lua -h
```

Train FSTN on the UCF101 dataset (101 classes) with 4 CPU threads (for data loading).  
Batch size is 128.  
Save models at every 5 epochs.
All GPUs is used for the training.
If you want a part of GPUs, use ```CUDA_VISIBLE_DEVICES=...```.

```bash
python main.py --root_path ~/data --video_path ucf101_videos/jpg \
--result_path results --dataset ucf101 --model FSTN \
--n_classes 101 --batch_size 128 --n_threads 4 --checkpoint 5
```

Continue Training from epoch 101. (~/data/results/save_100.pth is loaded.)

```bash
python main.py --root_path ~/data --video_path ucf101_videos/jpg \
--result_path results --dataset ucf101 --resume_path results/save_100.pth \
--n_classes 101 --batch_size 128 --n_threads 4 --checkpoint 5
```

Fine-tuning conv5_x and fc layers of a pretrained model (~/data/models/SDAN-50-kinetics.pth) on UCF-101.

```bash
python main.py --root_path ~/data --video_path ucf101_videos/jpg \
--result_path results --dataset ucf101 --n_classes 400 --n_finetune_classes 101 \
--pretrain_path models/SDAN-50-kinetics.pth --ft_begin_index 4 \
--model FSTN --batch_size 128 --n_threads 4 --checkpoint 5
```

