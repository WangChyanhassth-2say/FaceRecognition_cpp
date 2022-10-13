#include "Face.h"
#include <opencv2/dnn.hpp>


RecNet::RecNet():
        _target_size(112)
{
}

inline void RecNet::Release()
{
    cv::dnn::Net* Net = &recnet;
    if (Net != nullptr)
    {
        recnet.~Net();
        Net = nullptr;
    }
}

RecNet::RecNet(const char* model_path):
        _target_size(112)
{
    Init(model_path);
}

void RecNet::Init(const char* model_path)
{
    recnet = cv::dnn::readNetFromONNX(model_path);
}

void RecNet::Recognize(cv::Mat& bgr, bbox bbox, std::vector<float>& features)
{    
    Timer timer;
    timer.tic();

    features.clear();
    const int target_size = _target_size;
    int width = bgr.cols;
    int height = bgr.rows;
    float scale = 1.f;

    cv::Mat in = Process(bgr, bbox);
    cv::imshow("face", in);
    cv::waitKey(1);

    in = cv::dnn::blobFromImage(in);

    timer.toc("rec precoss:");

    timer.tic();

    recnet.setInput(in, "input");
    std::vector<float> out = recnet.forward("features");

    timer.toc("rec:");    

    std::vector<float> tmp;
    std::vector<float> *ptr = &out;//.ptr<float>(0);
    for(int i = 0; i < out.size(); ++i)
    {
        features.push_back((*ptr)[i]);
    }
}

inline void RecNet::SetDefaultParams()
{
    _target_size = 112;
}

RecNet::~RecNet()
{
    Release();
}

cv::Mat RecNet::MeanAxis0(const cv::Mat &src)
{
    int num = src.rows;
    int dim = src.cols;

    // x1 y1
    // x2 y2
    cv::Mat output(1,dim,CV_32F);
    for(int i = 0 ; i <  dim; i++){
        float sum = 0 ;
        for(int j = 0 ; j < num ; j++){
            sum+=src.at<float>(j,i);
        }
        output.at<float>(0,i) = sum/num;
    }

    return output;
}

cv::Mat RecNet::ElementwiseMinus(const cv::Mat &A,const cv::Mat &B)
{
    cv::Mat output(A.rows,A.cols,A.type());

    assert(B.cols == A.cols);
    if(B.cols == A.cols)
    {
        for(int i = 0 ; i <  A.rows; i ++)
        {
            for(int j = 0 ; j < B.cols; j++)
            {
                output.at<float>(i,j) = A.at<float>(i,j) - B.at<float>(0,j);
            }
        }
    }
    return output;
}

int RecNet::MatrixRank(cv::Mat M)
{
    cv::Mat w, u, vt;
    cv::SVD::compute(M, w, u, vt);
    cv::Mat1b nonZeroSingularValues = w > 0.0001;
    int rank = countNonZero(nonZeroSingularValues);
    return rank;

}

cv::Mat RecNet::VarAxis0(const cv::Mat &src)
{
    cv::Mat temp_ = ElementwiseMinus(src,MeanAxis0(src));
    cv::multiply(temp_ ,temp_ ,temp_ );
    return MeanAxis0(temp_);

}

cv::Mat RecNet::SimilarTransform(cv::Mat src,cv::Mat dst)
{
    int num = src.rows;
    int dim = src.cols;
    cv::Mat src_mean = MeanAxis0(src);
    cv::Mat dst_mean = MeanAxis0(dst);
    cv::Mat src_demean = ElementwiseMinus(src, src_mean);
    cv::Mat dst_demean = ElementwiseMinus(dst, dst_mean);
    cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
    cv::Mat d(dim, 1, CV_32F);
    d.setTo(1.0f);
    if (cv::determinant(A) < 0) {
        d.at<float>(dim - 1, 0) = -1;

    }
    cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
    cv::Mat U, S, V;

    cv::SVD::compute(A, S, U, V);

    int rank = MatrixRank(A);
    if (rank == 0) {
        assert(rank == 0);

    } else if (rank == dim - 1) {
        if (cv::determinant(U) * cv::determinant(V) > 0) {
            T.rowRange(0, dim).colRange(0, dim) = U * V;
        } else {

            int s = d.at<float>(dim - 1, 0) = -1;
            d.at<float>(dim - 1, 0) = -1;

            T.rowRange(0, dim).colRange(0, dim) = U * V;
            cv::Mat diag_ = cv::Mat::diag(d);
            cv::Mat twp = diag_*V; //np.dot(np.diag(d), V.T)
            cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
            cv::Mat C = B.diag(0);
            T.rowRange(0, dim).colRange(0, dim) = U* twp;
            d.at<float>(dim - 1, 0) = s;
        }
    }
    else{
        cv::Mat diag_ = cv::Mat::diag(d);
        cv::Mat twp = diag_*V.t(); //np.dot(np.diag(d), V.T)
        cv::Mat res = U* twp; // U
        T.rowRange(0, dim).colRange(0, dim) = -U.t()* twp;
    }
    cv::Mat var_ = VarAxis0(src_demean);
    float val = cv::sum(var_).val[0];
    cv::Mat res;
    cv::multiply(d,S,res);
    float scale =  1.0/val*cv::sum(res).val[0];
    T.rowRange(0, dim).colRange(0, dim) = - T.rowRange(0, dim).colRange(0, dim).t();
    cv::Mat temp1 = T.rowRange(0, dim).colRange(0, dim); // T[:dim, :dim]
    cv::Mat temp2 = src_mean.t(); //src_mean.T
    cv::Mat temp3 = temp1*temp2; // np.dot(T[:dim, :dim], src_mean.T)
    cv::Mat temp4 = scale*temp3;
    T.rowRange(0, dim).colRange(dim, dim+1)=  -(temp4 - dst_mean.t()) ;
    T.rowRange(0, dim).colRange(0, dim) *= scale;
    return T;
}

inline double RecNet::count_angle(float landmark[5][2])
{
    double a = landmark[2][1] - (landmark[0][1] + landmark[1][1]) / 2;
    double b = landmark[2][0] - (landmark[0][0] + landmark[1][0]) / 2;
    double angle = atan(abs(b) / a) * 180.0 / M_PI;
    return angle;
}

cv::Mat RecNet::Process(cv::Mat& SmallFrame, bbox& bbox)
{
    // gt face landmark
    float v1[5][2] = {
            {35.3437f, 51.6963f},
            {76.4538f, 51.5014f},
            {56.0294f, 71.7366f},
            {39.1409f, 92.3655f},
            {73.1849f, 92.2041f}
    };
    static cv::Mat src(5, 2, CV_32FC1, v1);
    memcpy(src.data, v1, 2*5*sizeof(float));

    // Perspective Transformation
    float v2[5][2] ={
        {bbox.point[0]._x, bbox.point[0]._y},
        {bbox.point[1]._x, bbox.point[1]._y},
        {bbox.point[2]._x, bbox.point[2]._y},
        {bbox.point[3]._x, bbox.point[3]._y},
        {bbox.point[4]._x, bbox.point[4]._y},
    };
    cv::Mat dst(5, 2, CV_32FC1, v2);
    memcpy(dst.data, v2, 2*5*sizeof(float));

    // lighter way
    // Angle = count_angle(v2);

    cv::Mat aligned = SmallFrame.clone();
    cv::Mat m = SimilarTransform(dst, src);
    cv::warpPerspective(SmallFrame, aligned, m, cv::Size(112, 112), cv::INTER_LINEAR);

    return aligned;
}