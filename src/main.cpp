#include <stdio.h>
#include <fstream>
#include <math.h>
#include <vector>
#include <algorithm>
#include "Face.h"
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#if CV_MAJOR_VERSION >= 3
#include "opencv2/video/video.hpp"
#endif

int main(int argc, char** argv)
{
    // cv::setNumThreads(16);
    
    const char* det_model_path = "../models/blazeface_simmir.onnx";
    //  const int det_side = 256;// able to pad // only for 128 times // pad may cost bugs
    const char* rec_model_path = "../models/mobilefacenet_simmir.onnx";
    // const int rec_side = 112;// not able to pad
    DetNet detector(det_model_path);
    RecNet recognizer(rec_model_path);
    Timer timer;

    // cv::VideoCapture cap = cv::VideoCapture(0);
    cv::Mat img;
    cv::Mat face;
    std::vector<bbox> boxes;
    std::vector<float> features;
    std::vector<float> gao2;
    std::vector<float> gao;
    std::vector<float> me;
    float score_gao;
    float score_gao2;
    float score_me;

    img = cv::imread("../images/gao2.jpg");
    cv::resize(img, img, cv::Size(256, 256));
    detector.Detect(img, boxes);
    recognizer.Recognize(img, boxes[0], gao2);

    img = cv::imread("../images/me.jpg");
    cv::resize(img, img, cv::Size(256, 256));
    detector.Detect(img, boxes);
    recognizer.Recognize(img, boxes[0], me);

    while (1)
    {
        // cap >> img;
        img = cv::imread("../images/gao4.jpg");
        cv::resize(img, img, cv::Size(256, 256));
        float scale = 1.f;
        if (img.empty())
        {
            return -1;
        }

        timer.tic();
        detector.Detect(img, boxes);

        for (int j = 0; j < boxes.size(); ++j) 
        {
            recognizer.Recognize(img, boxes[j], features);
            cv::Rect rect(boxes[j].x1/scale, boxes[j].y1/scale, boxes[j].x2/scale - boxes[j].x1/scale, boxes[j].y2/scale - boxes[j].y1/scale);
            cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
            // score_gao = vector_cos(features, gao);
            score_me = vector_cos(features, me);
            score_gao2 = vector_cos(features, gao2);
            char test[80];
            sprintf(test, "face:%.2f  me:%.2f  gao:%.2f", boxes[j].s, score_me, score_gao2);
            std::cout << test << std::endl;
            cv::putText(img, test, cv::Size((boxes[j].x1/scale), boxes[j].y1/scale), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 0));
            cv::circle(img, cv::Point(boxes[j].point[0]._x / scale, boxes[j].point[0]._y / scale), 1, cv::Scalar(0, 0, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[1]._x / scale, boxes[j].point[1]._y / scale), 1, cv::Scalar(0, 255, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[2]._x / scale, boxes[j].point[2]._y / scale), 1, cv::Scalar(255, 0, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[3]._x / scale, boxes[j].point[3]._y / scale), 1, cv::Scalar(0, 255, 0), 4);
            cv::circle(img, cv::Point(boxes[j].point[4]._x / scale, boxes[j].point[4]._y / scale), 1, cv::Scalar(255, 0, 0), 4);
        }
        timer.toc("-----------total cost");
        cv::imshow("all",img);
        cv::waitKey(1);

    
    }
    return 0;
}

