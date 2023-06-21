# DeepFake
Comparison of a few DeepFake Detection Algorithms


## If you want to work on Jupyter Notebook check: [DeepFake Detection Comparison Notebook](DeepFake_Detection_Model_Comparison.ipynb)


## Downloading Data

First of all, you must download the dataset, go to [README](data/README.txt) and download dataset from link!


## Creating Environment

First, you need to create a new virtual env with: 
```sh
python -m venv "/path/to/new/virtual/environment" 
```
Then, install requirements.txt to setup environment:
```sh
pip install -r requirements.txt
```


## Testing Selim EfficientNet B7 (CNN)

First, you need to download pretrained models: Go to [README](CNN/weights/README.txt)


Then, run this script with arguments you want: 
```sh
python CNN/predict_folder.py --models "downloaded model names" --test-dir "videos path"
```
Example:
```sh
python CNN/predict_folder.py --models "final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36" --test-dir "./data/0"
```

After testing process is finished, run this script for results:
```sh
python extract_results.py --path "./CNN/submission.csv"
```


## Testing Cross. ViT EfficientNet B0 (ViT)

First, you need to download pretrained model: go to [README](ViT/README.txt)


Then, run this scripts to preprocess videos:

1.
```sh
python ViT/preprocessing/detect_faces.py --data_path "videos path"
```
Example:
```sh
python ViT/preprocessing/detect_faces.py --data_path "data/0"
```

2.
```sh
python ViT/preprocessing/extract_crops.py --data_path "videos path" --output_path "folder for frames"
```
Example:
```sh
python ViT/preprocessing/extract_crops.py --data_path "data/0" --output_path "data/frames"
```


After preprocessing, run this script to test the model:
```sh
python ViT/test.py --model_path "model path" --efficient_net "EfficientNet Model 0-7 (int)" --frames_per_video "frames per video (int)" 
```
Example:
```sh
python ViT/test.py --model_path "ViT/cross_efficient_vit.pth" --efficient_net 0 --frames_per_video 32
```

After testing process is finished, run this script for results:
```sh
python extract_results.py --path "./ViT/out/DFDC/submission.csv"
```
