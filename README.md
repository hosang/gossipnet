# Learning non-maximum suppression for object detection

This is the code for the paper  
_Learning non-maximum suppression. Jan Hosang, Rodrigo Benenson, Bernt Schiele. CVPR 2017._

You can find the project page with downloads here: https://mpi-inf.mpg.de/learning-nms

## Setup
need to link to the coco API in the root directory, like so:
```
/work/src/tf-gnet$ ln -s /work/src/coco/PythonAPI/pycocotools
```

need to link to coco annotations/images in the data subdir:
```
/work/src/tf-gnet/data$ ln -s /datasets/coco
```

