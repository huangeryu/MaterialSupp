#ifndef __BRAND_CLASSIFIER_H__
#define __BRAND_CLASSIFIER_H__
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
using namespace std;
using namespace caffe;
using namespace cv;
class BrandClassifier
{
public:
    BrandClassifier(string cfg_file,string weight_file,string mean_file);
    void warpper();
    int predict();
    int loadImage(string img_file);
    int analyzeResult(float* max);
private:
    vector<cv::Mat> input;
    cv::Mat mean_img;
    std::shared_ptr<Net<float>> net;
};

#endif