# install thirdparties from local clones

# detectron2
git clone https://github.com/facebookresearch/detectron2.git thirdparty
python -m pip install -e thirdparty/detectron2

# segment_anything
git clone https://github.com/facebookresearch/segment-anything.git thirdparty
pip install -e thirdparty/segment-anything

# hloc
git clone --recursive https://github.com/cvg/Hierarchical-Localization/ thirdparty
python -m pip install -e thirdparty/Hierarchical-Localization