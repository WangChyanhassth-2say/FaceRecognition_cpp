#ifndef FACE_H
#define FACE_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <stdio.h>
#include <stack>
#include <chrono>
using namespace std::chrono;

class Timer
{
// My Timer Implement
// 
// function tic is for logging start timestamp
// and the function toc is for printing
//
// It's just a simple implement, but useful enough for release version
// meeting bugs while waiting such as cv::waitKey(0) or input
public:
    std::stack<high_resolution_clock::time_point> tictoc_stack;
    
    void tic()
    {
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        tictoc_stack.push(t1);
    }

    double toc(std::string msg = "", bool flag = true)
    {
        double diff = duration_cast<milliseconds>(high_resolution_clock::now() - tictoc_stack.top()).count();
        if(msg.size() > 0){
            if (flag)
                printf("%s time elapsed: %f ms\n", msg.c_str(), diff);
        }

        tictoc_stack.pop();
        return diff;
    }

    void reset()
    {
        tictoc_stack = std::stack<high_resolution_clock::time_point>();
    }
};

inline float vector_dot(const std::vector<float>& feat1, const std::vector<float>& feat2)
{
    float res = 0.f;
    for(size_t i = 0; i < feat1.size(); ++i)
    {
        res += feat1[i] * feat2[i];
    }
    return res;
}

inline float vector_cos(const std::vector<float>& feat1, const std::vector<float>& feat2)
{
    // feat1 * feat2 / (||feat1|| * ||feat2||)
    return vector_dot(feat1, feat2) / (sqrt(vector_dot(feat1, feat1)) * sqrt(vector_dot(feat2, feat2)));
}

struct Point
{
    float _x;
    float _y;
};

struct bbox
{
    float x1;
    float y1;
    float x2;
    float y2;
    float s;
    Point point[5];
};

struct box
{
    float cx;
    float cy;
    float sx;
    float sy;
};

class DetNet
{
// My DetNet Implement
// 
// see doc
public:
    DetNet();
    DetNet(const char* model_path, const int target_size=256);
    void Init(const char* model_path);
    inline void SetDefaultParams();
    inline void Release();
    void nms(std::vector<bbox> &input_boxes, float NMS_THRESH);
    void Detect(cv::Mat& bgr, std::vector<bbox>& boxes);
    void create_anchor(std::vector<box> &anchor, int w, int h);
    static inline bool cmp(bbox a, bbox b); // for nms
    ~DetNet();

public:
    float _nms;
    float _threshold;
    float _mean_val[3];
    int _target_size;
    cv::dnn::Net detnet;
};

class RecNet
{
// My RecNet Implement
// 
// see doc
public:
    RecNet();
    RecNet(const char* model_path, const int target_size=112);
    void Init(const char* model_path);
    inline void SetDefaultParams();
    inline void Release();
    void Recognize(cv::Mat& bgr, bbox bbox, std::vector<float>& features);
    cv::Mat SimilarTransform(cv::Mat src,cv::Mat dst);
    cv::Mat VarAxis0(const cv::Mat &src);
    int MatrixRank(cv::Mat M);
    cv::Mat ElementwiseMinus(const cv::Mat &A,const cv::Mat &B);
    cv::Mat MeanAxis0(const cv::Mat &src);
    cv::Mat Process(cv::Mat& SmallFrame, bbox& bbox);
    inline double count_angle(float landmark[5][2]); // lighter way
    ~RecNet();

public:
    bbox _bbox_detected;
    int _target_size;
    cv::dnn::Net recnet;
};
#endif 
