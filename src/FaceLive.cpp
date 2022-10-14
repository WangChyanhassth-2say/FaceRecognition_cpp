#include "Face.h"
#include <algorithm>
#include <opencv2/dnn.hpp>


LiveNet::LiveNet():
        _target_size(80),
        _threshold(0.1)
{
}

inline void LiveNet::Release()
{
    cv::dnn::Net* Net = &livenet;
    if (Net != nullptr)
    {
        livenet.~Net();
        Net = nullptr;
    }
}

LiveNet::LiveNet(const char* model_path):
        _target_size(80),
        _threshold(0.1)
{
    Init(model_path);
}

void LiveNet::Init(const char* model_path)
{
    livenet = cv::dnn::readNetFromONNX(model_path);
}

inline std::vector<float> LiveNet::softmax(std::vector<float> input)
{
    float total = 0.;
    for(auto x : input)
    {
        total += exp(x);
    }
    std::vector<float> result;
    for(auto x : input)
    {
        result.push_back(exp(x) / total);
    }
    return result;
}

void LiveNet::LiveCheck(cv::Mat& bgr, bbox bbox, checkbox& livecheckbox)
{    
    Timer timer;
    timer.tic();

    livecheckbox = {true, 1.};

    cv::Mat in = Process(bgr, bbox);
    // cv::imshow("live", in);
    // cv::waitKey(1);

    in = cv::dnn::blobFromImage(in);
    timer.toc("live precoss:");

    timer.tic();

    livenet.setInput(in, "input");
    std::vector<float> out = livenet.forward("scores");

    timer.toc("live:");

    std::vector<float> tmp;
    out = this->softmax(out);
	auto valueIter = max_element(out.begin(), out.end());
	int index = distance(out.begin(), valueIter);

    int doublecheck = 0;
    livecheckbox.isReal = (index == 1) ? 1 : 0;
    livecheckbox.conf = *(out.begin()+1);
    if (livecheckbox.conf >= _threshold)
    {
        doublecheck = 1;
    }
    if (livecheckbox.isReal != doublecheck)
    {
        livecheckbox.isReal = 2;
    }
}

inline void LiveNet::SetDefaultParams()
{
    _target_size = 80;
    _threshold = 0.1;
}

LiveNet::~LiveNet()
{
    Release();
}

inline double LiveNet::count_angle(float landmark[2][2])
{
    double dy = (landmark[1][1] - landmark[0][1]);
    double dx = (landmark[1][0] - landmark[0][0]);
    double angle = atan2(dy, dx) * 180.0/CV_PI;
    return angle;
}

cv::Mat LiveNet::Process(cv::Mat& bgr, bbox& bbox)
{
    // WrapAffine with two-eyes-angle
    float v2[2][2] ={
        {bbox.point[0]._x, bbox.point[0]._y},
        {bbox.point[1]._x, bbox.point[1]._y}
    };
    cv::Mat dst(2, 2, CV_32FC1, v2);
    memcpy(dst.data, v2, 2*2*sizeof(float));

    cv::Mat rot_mat = cv::getRotationMatrix2D(
        cv::Point2f((bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2),
        count_angle(v2),
        1.0);
    cv::Mat aligned = bgr.clone();
    cv::warpAffine(aligned, aligned, rot_mat, cv::Size(bgr.cols, bgr.rows));

    // Expand the bbox range that fits the livechecker model
    int bboxwidth = bbox.x2 - bbox.x1;
    int bboxheight = bbox.y2 - bbox.y1;
    int y1 = (bbox.y1 - bboxheight > 0) ? (bbox.y1 - bboxheight) : 0;
    int y2 = (bbox.y2 + bboxheight < bgr.rows) ? (bbox.y2 + bboxheight) : bgr.rows;
    int x1 = (bbox.x1 - bboxwidth  > 0) ? (bbox.x1 - bboxwidth) : 0;
    int x2 = (bbox.x2 + bboxwidth  < bgr.cols) ? (bbox.x1 + bboxwidth) : bgr.cols;

    cv::resize(aligned(cv::Range(y1, y2), cv::Range(x1, x2)), aligned, cv::Size(80, 80));

    return aligned;
}