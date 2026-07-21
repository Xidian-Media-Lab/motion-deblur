# Motion Deblurring of Long Exposure Images in Low Light Condition by Selective Fusion of Visible and Near Infrared Images
<img width="1327" height="967" alt="GA" src="https://github.com/user-attachments/assets/d5f7e78b-4f0a-482b-abe0-4a096e6b1a5d" />

## Enviroments
- Python=3.8.12
- Pytorch=1.10.2
- numpy=1.21.2
- matlab R2018b (only for evaluation)

## How to Use
## Test
python correction.py \
python test.py

## Train
python train.py

## Evaluation
function [SF, Qabf, VIF, EN, MI, FMI_pixel, FMI_dct, FMI_w] = fusion_metrics(image_f,image_nir,image_vis)
