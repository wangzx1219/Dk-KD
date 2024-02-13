## Requirements and Installation
* python >= 3.7
* pytorch >= 1.10.0
* faiss-gpu >= 1.7.3
* sacremoses == 0.0.41
* sacrebleu == 1.5.1
* fastBPE == 0.1.0
* streamlit >= 1.13.0
* scikit-learn >= 1.0.2
* seaborn >= 0.12.1

Installation and configuration of the environment:
```shell
git clone https://github.com/wangzx1219/Dk-KD.git
cd Dk-KD
pip install --editable ./
```

Installing faiss :

```bash
conda install faiss-gpu -c pytorch 
```

## Data Preparation

prepare pretrained models and dataset:

```bash
cd knnbox-scripts
bash prepare_dataset_and_model.sh
```

## Train

train teacher model and final model

```bash
cd Dk-KD_scripts
bash train_teacher.sh
bash train_final.sh
```

## Citation


```bibtex

```
