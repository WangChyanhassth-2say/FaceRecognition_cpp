#include "Face.h"
#include <opencv2/dnn.hpp>


DetNet::DetNet():
        _nms(0.5),
        _threshold(0.65),
        _mean_val{104.f, 117.f, 123.f},
        _target_size(256)
{
}

inline void DetNet::Release()
{
        // delete detnet; // TODO
}

DetNet::DetNet(const char* model_path, const int target_size):
        _nms(0.5),
        _threshold(0.65),
        _mean_val{104.f, 117.f, 123.f},
        _target_size(target_size)
{
    Init(model_path);
}

void DetNet::Init(const char* model_path)
{
    detnet = cv::dnn::readNetFromONNX(model_path);
}

void DetNet::Detect(cv::Mat& bgr, std::vector<bbox>& boxes)
{    
    Timer timer;
    timer.tic();

    boxes.clear();
    const int target_size = _target_size;
    int width = bgr.cols;
    int height = bgr.rows;
    // letterbox pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;

    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    cv::Mat in;
    cv::resize(bgr, in, cv::Size(w, h));
    int wpad = (w + 15) / 16 * 16 - w;
    int hpad = (h + 15) / 16 * 16 - h;

    cv::Mat in_pad;
    cv::copyMakeBorder(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, cv::BORDER_CONSTANT, 114.f);

    in_pad = cv::dnn::blobFromImage(in_pad);

    timer.toc("det precoss:");

    timer.tic();

    detnet.setInput(in_pad, "input");
    cv::Mat out = detnet.forward("boxes");
    cv::Mat out1 = detnet.forward("scores");
    cv::Mat out2 = detnet.forward("landmarks");

    timer.toc("det:");

    std::vector<box> anchor;
    timer.tic();
    
    create_anchor(anchor, w, h);
    timer.toc("anchor:");

    std::vector<bbox > total_box;
    float *ptr = out.ptr<float>(0);
    float *ptr1 = out1.ptr<float>(0);
    float *landms = out2.ptr<float>(0);

    for (int i = 0; i < anchor.size(); ++i)
    {   
        if (*(ptr1+1) > _threshold)
        {
            box tmp = anchor[i];
            box tmp1;
            bbox result;

            // loc and conf
            tmp1.cx = tmp.cx + *ptr * 0.1 * tmp.sx;
            tmp1.cy = tmp.cy + *(ptr+1) * 0.1 * tmp.sy;
            tmp1.sx = tmp.sx * exp(*(ptr+2) * 0.2);
            tmp1.sy = tmp.sy * exp(*(ptr+3) * 0.2);

            result.x1 = (tmp1.cx - tmp1.sx/2) * in.cols;
            if (result.x1<0)
                result.x1 = 0;
            result.y1 = (tmp1.cy - tmp1.sy/2) * in.rows;
            if (result.y1<0)
                result.y1 = 0;
            result.x2 = (tmp1.cx + tmp1.sx/2) * in.cols;
            if (result.x2>in.cols)
                result.x2 = in.cols;
            result.y2 = (tmp1.cy + tmp1.sy/2)* in.rows;
            if (result.y2>in.rows)
                result.y2 = in.rows;
            result.s = *(ptr1 + 1);

            // landmark
            for (int j = 0; j < 5; ++j)
            {
                result.point[j]._x =( tmp.cx + *(landms + (j<<1)) * 0.1 * tmp.sx ) * in.cols;
                result.point[j]._y =( tmp.cy + *(landms + (j<<1) + 1) * 0.1 * tmp.sy ) * in.rows;
            }
            total_box.push_back(result);
        }
        
        ptr += 4;
        ptr1 += 2;
        landms += 10;
    }

    std::sort(total_box.begin(), total_box.end(), cmp);
    nms(total_box, _nms);

    for (int j = 0; j < total_box.size(); ++j)
    {   
        total_box[j].x1 = (total_box[j].x1 - (wpad / 2)) / scale;
        total_box[j].y1 = (total_box[j].y1 - (hpad / 2)) / scale;
        total_box[j].x2 = (total_box[j].x2 - (wpad / 2)) / scale;
        total_box[j].y2 = (total_box[j].y2 - (hpad / 2)) / scale;

        for (int k = 0; k < 5; ++k)
            {
                total_box[j].point[k]._x = (total_box[j].point[k]._x  - (wpad / 2)) / scale;
                total_box[j].point[k]._y = (total_box[j].point[k]._y  - (hpad / 2)) / scale;
            }
        boxes.push_back(total_box[j]);
    }
}

inline bool DetNet::cmp(bbox a, bbox b) 
{
    if (a.s > b.s)
        return true;
    return false;
}

inline void DetNet::SetDefaultParams()
{
    _nms = 0.5;
    _threshold = 0.65;
    _mean_val[0] = 104;
    _mean_val[1] = 117;
    _mean_val[2] = 123;
}

DetNet::~DetNet()
{
    Release();
}

void DetNet::create_anchor(std::vector<box> &anchor, int w, int h)
{
    anchor.clear();
    std::vector<std::vector<int> > feature_map(2), min_sizes(4);
    float steps[] = {8, 16};
    for (int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h/steps[i]));
        feature_map[i].push_back(ceil(w/steps[i]));
    }
    std::vector<int> minsize1 = {8, 11};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {14, 19, 26, 38, 64, 149};
    min_sizes[1] = minsize2;


    for (int k = 0; k < feature_map.size(); ++k)
    {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i)
        {
            for (int j = 0; j < feature_map[k][1]; ++j)
            {
                for (int l = 0; l < min_size.size(); ++l)
                {
                    float s_kx = min_size[l]*1.0/w;
                    float s_ky = min_size[l]*1.0/h;
                    float cx = (j + 0.5) * steps[k]/w;
                    float cy = (i + 0.5) * steps[k]/h;
                    box axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                }
            }
        }

    }

}

void DetNet::nms(std::vector<bbox> &input_boxes, float NMS_THRESH)
{
    std::vector<float>vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}


