# FaceRecognition_cpp
face detect, align and recognize via onnx models implemented with cpp

## Quick Strat  
### // for installing  
ensure you have cmake and makefile  
### // for compiling  
ensure you have opencv in your env  
onnx lib is provided in ./3rdparty/  

use the command    
```
mkdir build  
cd build  
cmake ..  
make  
./main  
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
we choose the front model only, which needs the input shape with [1, 3, 256, 256]  
tarined on 5-keypoints dataset, which is easier to align than mediapipe provided 6-keypoints verion  
by the way, we're not using pre-anchor here for speed  
```
inputs:  
    name: input  
    type: float32[1,3,256,256]  
    
outputs:  
    name: boxes  
    type: float32[1,896,4]  
 
    name: scores  
    type: float32[1,896,2]  
  
    name: landmarks  
    type: float32[1,896,10]  
```

#### // for human face features recognition  
we use MobileFaceNet-v2 model  
at the front layers there are layers of img preprocessing  
before the output of dims 512, which is too heavy for us  
we reshaped once, make the features keep in 128 dims  
```
inputs:  
    name: input  
    type: float32[1,3,112,112]  
    
outputs:  
    name: features  
    type: float32[1,128]  
```

#### // for cpp implement details  
in src you can see Face.h, FaceDet.cpp and FaceRec.cpp  

##### Face.h  
in Face.h we defined:  

    class Timer for logging the timestamp  
    
    two inline function that got vector cosine  
    
    some structs  
    
    class DetNet for face detect  
    
    class RecNet for face features recognition  

##### FaceDet.cpp
in FaceDet.cpp we implemented:  
```
DetNet initial funtion   
    # needs const char* model_path inputs  

Detect function  
    # needs cv::Mat bgr_image and a std::vector<bbox> output container  
    # in this function we clear the output vector first  
    # then we got bgr_image resize, pad and bolb, sending it into the model  
    # no imgNorm here because we may add the liveness detector later  
    # after post process we got face bbox with its location, confidence and 5-points info  
    # we got a loop for multi-faces, so the output is std::vector<bbox>, means multi bboxes  
    
create_anchor, nms function and compare function  
    #to conver the model outputs to bboxes format  
```

##### FaceRec.cpp
in FaceRec.cpp we implemented:  
```
RecNet initial funtion   
    # needs const char* model_path inputs  

Detect function  
    # needs cv::Mat bgr_image, this guy's bbox, and a std::vector<float> feature container for him  
    # in this function we clear the features vector first  
    # then we got bgr_image align with the face, crop and blob, sending it into the model  
    # directly returns the features of the man in the bbox, just one guy  
    
the functions remain  
    # to make the face align using the 5-points info  
    # we provide two ways to finish that  
    # similar transform is a little bit heavier, but robuster with the yaw and pitch rotations   
    # while we also able to align with the angle between the line of two eyes and the horizen  
```

TODO: 
- [x] remove target_size input, make it stable  
- [x] fix bugs that pad in face detect will reduce the performance   
- [x] provide imgNorm function  
- [x] provide Release function  
- [ ] provide liveness detector  
- [ ] provide opencv libs
