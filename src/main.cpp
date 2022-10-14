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
    // cv::setNumThreads(1);
    
    const char* det_model_path = "../models/FaceDet.onnx";
    const char* rec_model_path = "../models/FaceRec.onnx";
    const char* live_model_path = "../models/FaceLive.onnx";
    DetNet detector(det_model_path);
    RecNet recognizer(rec_model_path);
    LiveNet livechecker(live_model_path);
    Timer timer;

    cv::Mat img;
    cv::Mat face;
    std::vector<bbox> boxes;
    std::vector<float> features;
    std::vector<float> gao2;
    std::vector<float> gao;
    std::vector<float> me;
    checkbox livecheckbox;
    float score_gao;
    float score_gao2;
    float score_me;
    std::vector<const char*> flags = {"Fake", "Real", "Maybe"};
    
    img = cv::imread("../images/gao.jpg");
    cv::resize(img, img, cv::Size(1024, 1024));
    detector.Detect(img, boxes);
    recognizer.Recognize(img, boxes[0], gao2);

    img = cv::imread("../images/me.jpg");
    cv::resize(img, img, cv::Size(1024, 1024));    
    detector.Detect(img, boxes);
    recognizer.Recognize(img, boxes[0], me);

    cv::VideoCapture cap = cv::VideoCapture(0);
    int count = 0;
    while (1)
    {
        count += 1;
        if(count > 500)
        {
            detector.~DetNet();
            livechecker.~LiveNet();
            recognizer.~RecNet();
            break;
        }
        cap >> img;
        cv::resize(img, img, cv::Size(560, 315));
        if (img.empty())
        {
            return -1;
        }

        timer.tic();
        detector.Detect(img, boxes);

        for (int j = 0; j < boxes.size(); ++j) 
        {
            livechecker.LiveCheck(img, boxes[j], livecheckbox);
            recognizer.Recognize(img, boxes[j], features);
            cv::Rect rect(boxes[j].x1, boxes[j].y1, boxes[j].x2 - boxes[j].x1, boxes[j].y2 - boxes[j].y1);
            cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
            score_me = vector_cos(features, me);
            score_gao2 = vector_cos(features, gao2);
            char test[80];
            char livescore[80];
            sprintf(test, "face:%.2f  me:%.2f  gao:%.2f", boxes[j].s, score_me, score_gao2);
            sprintf(livescore, "%s  %.2f", flags[livecheckbox.isReal], livecheckbox.conf);

            cv::putText(img, test, cv::Size((boxes[j].x1), boxes[j].y1), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 0));
            cv::putText(img, livescore, cv::Size((boxes[j].x1), boxes[j].y1-20), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 0));
            cv::circle(img, cv::Point(boxes[j].point[0]._x, boxes[j].point[0]._y), 1, cv::Scalar(0, 0, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[1]._x, boxes[j].point[1]._y), 1, cv::Scalar(0, 255, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[2]._x, boxes[j].point[2]._y), 1, cv::Scalar(255, 0, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[3]._x, boxes[j].point[3]._y), 1, cv::Scalar(0, 255, 0), 4);
            cv::circle(img, cv::Point(boxes[j].point[4]._x, boxes[j].point[4]._y), 1, cv::Scalar(255, 0, 0), 4);
        }
        timer.toc("-----------total cost");
        cv::imshow("all",img);
        cv::waitKey(1);

    }
    return 0;
}

