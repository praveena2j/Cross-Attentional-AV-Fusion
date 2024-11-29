# Cross Attentional Audio-Visual Fusion for Dimensional Emotion Recognition
Code for our paper "Cross Attentional Audio-Visual Fusion for Dimensional Emotion Recognition" accepted to IEEE FG 2021. Our paper can be found [here](https://ieeexplore.ieee.org/abstract/document/9667055).

## Citation

If you find this code useful for your research, please cite our paper.

```
@INPROCEEDINGS{9667055,
  author={Praveen, R. Gnana and Granger, Eric and Cardinal, Patrick},
  booktitle={2021 16th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2021)}, 
  title={Cross Attentional Audio-Visual Fusion for Dimensional Emotion Recognition}, 
  year={2021},
 }
```

This code uses the RECOLA dataset to validate the proposed approach for Dimensional Emotion Recognition. There are three major blocks in this repository to reproduce the results of our paper. This code uses Mixed Precision Training (torch.cuda.amp). The dependencies and packages required to reproduce the environment of this repository can be found in the `environment.yml` file. 

### Creating the environment
Create an environment using the `environment.yml` file

`conda env create -f environment.yml`

### Models
The pre-trained models of audio backbones are obtained [here](https://drive.google.com/file/d/1UA4mUB0XPICm8tFiIANMvDCBff4ZxLMT/view?usp=sharing)

The pre-trained models of visual backbones are obtained [here](https://drive.google.com/file/d/1KmeqxY2eJ-vb-wIi-a0gQGGjNR_E9K1-/view?usp=sharing)

The fusion models trained using our fusion approach can be found [here](https://drive.google.com/file/d/1invmOyC4dfkKb9_HaGpLk6o933fJBGG4/view?usp=sharing)

```
audiomodel.t7:  Visual model trained using RECOLA dataset
visualmodel.t7:  Audio model trained using RECOLA dataset
cam_model.pt:  Fusion model trained using our approach on the RECOLA dataset
```

# Table of contents <a name="Table_of_Content"></a>

+ [Preprocessing](#DP) 
    + [Step One: Download the dataset](#PD)
    + [Step Two: Preprocess the visual modality](#PV) 
    + [Step Three: Preprocess the audio modality](#PA)
    + [Step Four: Preprocess the annotations](#PL)
+ [Training](#Training) 
    + [Training the fusion model](#TE) 
+ [Inference](#R)
    + [Generating the results](#GR)
 
## Preprocessing <a name="DP"></a>
[Return to Table of Content](#Table_of_Content)

### Step One: Download the dataset <a name="PD"></a>
[Return to Table of Content](#Table_of_Content)
Please download the following.
  + The dataset can be downloaded [here]([https://ibug.doc.ic.ac.uk/resources/aff-wild2/](https://diuf.unifr.ch/main/diva/recola/download.html))

### Step Two: Preprocess the visual modality <a name="PD"></a>
[Return to Table of Content](#Table_of_Content)
  + You may choose to use [OpenFace toolkit](https://github.com/TadasBaltrusaitis/OpenFace/releases) to extract the cropped-aligned images.

### Step Three: Preprocess the audio modality <a name="PD"></a>
[Return to Table of Content](#Table_of_Content)
  + The audio files are extracted and segmented to generate the corresponding audio files in alignment with the visual files using [mkvextract](https://mkvtoolnix.download/). 

### Step Four: Preprocess the annotations <a name="PD"></a>
[Return to Table of Content](#Table_of_Content)
  + The annotations provided by the dataset organizers are preprocessed to obtain the labels of aligned audio and visual files. 

## Training <a name="DP"></a>
[Return to Table of Content](#Table_of_Content)
  + After obtaining the preprocessed audio and visual files along with annotations, we can train the model using the proposed fusion approach using the main.py script.

## Inference <a name="DP"></a>
[Return to Table of Content](#Table_of_Content)
  + The results of the proposed model can be reproduced using the trained model.

