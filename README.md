# FaceRecognition_cpp
face detect, align and recognize via onnx models implemented with cpp

## Quick Strat  
### // for installing  
ensure you have cmake and makefile  
### // for compiling  
ensure you have opencv in your env  
onnx lib is provided in ./3rdparty/  

use the command for install  
```
mkdir build  
cd build  
cmake ..  
make && ./main  
```

## Details
### // for folders  
./3rdparty is where the 3rd party libs stay  
./images is where test images are  
./models is where the onnx models are  
./src is where the source files are  

### // for models  
#### // for human face detect  
we provide reimplemented Google BlazeFace-Front model  
it's a lightweight model that easy to deploy and extremely fast  
we choose the front model only, which needs the input shape with [1, 3, 128, 128]  
tarined on 5-keypoints dataset, which is easier to align than mediapipe provided 6-keypoints verion  
by the way, we're not using pre-anchor here for speed  

#### // for human face features recognition  
we use MobileFaceNet-v2 model  
before the output of dims 512, we think its too heavy  
so we reshaped once before output, make the features keep in 128 dims  

TODO: 
- [ ] provide opencv lib
