# Applied Statistics and Machine Learning 

## Project Description
Demonstration of Machine Learning Models on a Mobile Device.

App will contain two machine learning models. 
The first classifies goats into diseased versus non-diseased based on a photograph of their eye.
The second transcribes handwritten text based on a photograph of the text.

The machine learning models themselves are too large to include in GitHub Repo. 
Download the models from the links below and include in the same directory as the python script.

---

## Authors 

Professors: Dr. Sun, Auburn University, and Dr. Chen, Tuskeegee University

GTA: Sam Chamoun

Students: Griffin Smith, Duncan Selle, Joshua Paulsen, and Adelina Rwabilimbo

---

## Installation & Run Instructions

1. Clone the project to a local repository<br>
`Git clone https://github.com/T-Griffin-Smith/ASML.git`

2. Download the machine learning models from the direct download links below.<br>Put the files in the same directory as ASML.py.
| Model | Download Link |
|  ---  |      ---      |
|GoatEyesModel.pth | https://drive.google.com/uc?export=download&id=1UrEeddj0_lEA3zVMyF0LRUeDoM2WtOFc |
|TextOCRModel.pth | https://drive.google.co
m/uc?export=download&id=1tj7lJm61PhHtQDHlCBRNustfZRscDUeI|
3. Download necessary python packages. The (very big) command below will install all necessary packages with the correct versions:<br>
`pip install \
absl-py==2.2.2 \
asttokens==3.0.0 \
astunparse==1.6.3 \
buildozer==1.5.0 \
certifi==2025.1.31 \
charset-normalizer==3.4.1 \
colorama==0.4.6 \
contourpy==1.3.1 \
cycler==0.12.1 \
decorator==5.2.1 \
distlib==0.3.9 \
docutils==0.21.2 \
efficientnet_pytorch==0.7.1 \
executing==2.2.0 \
filelock==3.18.0 \
filetype==1.2.0 \
flatbuffers==25.2.10 \
fonttools==4.57.0 \
fsspec==2025.3.2 \
gast==0.6.0 \
google-pasta==0.2.0 \
grpcio==1.71.0 \
h5py==3.13.0 \
huggingface-hub==0.30.2 \
idna==3.10 \
ipython==9.2.0 \
ipython_pygments_lexers==1.1.1 \
jedi==0.19.2 \
Jinja2==3.1.6 \
joblib==1.4.2 \
kagglehub==0.3.12 \
keras==3.9.2 \
Kivy==2.3.1 \
kivy-deps.angle==0.4.0 \
kivy-deps.glew==0.3.1 \
Kivy-Garden==0.1.5 \
kivy_deps.sdl2==0.8.0 \
kiwisolver==1.4.8 \
libclang==18.1.1 \
Markdown==3.8 \
markdown-it-py==3.0.0 \
MarkupSafe==3.0.2 \
matplotlib==3.10.1 \
matplotlib-inline==0.1.7 \
mdurl==0.1.2 \
ml_dtypes==0.5.1 \
mpmath==1.3.0 \
munch==4.0.0 \
namex==0.0.9 \
networkx==3.4.2 \
numpy==2.1.3 \
opencv-python==4.11.0.86 \
opt_einsum==3.4.0 \
optree==0.15.0 \
packaging==24.2 \
pandas==2.2.3 \
parso==0.8.4 \
pexpect==4.9.0 \
pillow==11.2.1 \
platformdirs==4.3.7 \
pretrainedmodels==0.7.4 \
prompt_toolkit==3.0.51 \
protobuf==5.29.4 \
ptyprocess==0.7.0 \
pure_eval==0.2.3 \
Pygments==2.19.1 \
pyparsing==3.2.3 \
pypiwin32==223 \
python-dateutil==2.9.0.post0 \
pytz==2025.2 \
pywin32==310 \
PyYAML==6.0.2 \
requests==2.32.3 \
rich==14.0.0 \
safetensors==0.5.3 \
scikit-learn==1.6.1 \
scipy==1.15.2 \
seaborn==0.13.2 \
segmentation_models_pytorch==0.4.0 \
sh==2.2.2 \
six==1.17.0 \
stack-data==0.6.3 \
sympy==1.13.1 \
tensorboard==2.19.0 \
tensorboard-data-server==0.7.2 \
tensorflow==2.19.0 \
tensorflow-io-gcs-filesystem==0.31.0 \
termcolor==3.0.1 \
threadpoolctl==3.6.0 \
timm==1.0.15 \
torch==2.6.0 \
torchvision==0.21.0 \
tqdm==4.67.1 \
traitlets==5.14.3 \
typing_extensions==4.13.2 \
tzdata==2025.2 \
urllib3==2.4.0 \
virtualenv==20.30.0 \
wcwidth==0.2.13 \
Werkzeug==3.1.3 \
wrapt==1.17.2`

4. Run `ASML.py`

---

## Project Successes

The app created with Kivy works well.

The buttons are bound to python function definitions for loading an image, selecting a model, and passing the image to the selected model. The result is then displayed on-screen. This process works perfectly for GoatEyesModel.

The training results showed good CTC losses for TextOCRModel. Unfortunately, the model did not work beyond this, as described below.

---

## Known Issues


The project is known to have the following issues:

- The Text Classification model does not work. The CTC layer was not saved and `model.predict()` attempts to re-train the model. 
- Need to find an alternative to PyTorch for GoatEyesModel. The model itself is 150 MB, and the Buildozer APK is much too large.
