conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y
python -m pip install -e detectron2
pip install fvcore==0.1.2.post20201122
pip install git+https://github.com/cocodataset/panopticapi.git
python -m pip install cityscapesscripts
python setup.py build develop
pip install opencv-python
