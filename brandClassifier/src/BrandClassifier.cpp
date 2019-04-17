#include "BrandClassifier.h"
#include <algorithm>
#include<iostream>
BrandClassifier::BrandClassifier(string cfg_file,string weight_file,string mean_file)
{
    Caffe::set_mode(Caffe::GPU);
    net.reset(new Net<float>(cfg_file,TEST));
    net->CopyTrainedLayersFromBinaryProto(weight_file);
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file,&blob_proto);
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    vector<Mat> channels;
    float* data=mean_blob.mutable_cpu_data();
    for(int i=0;i<mean_blob.channels();i++)
    {
        Mat channel(mean_blob.height(),mean_blob.width(),CV_32FC1,data);
        channels.push_back(channel);
        data+=mean_blob.height()*mean_blob.width();
    }
    Mat mean;
    cv::merge(channels,mean);
    cv::Scalar channel_mean=cv::mean(mean);
    mean_img=Mat(mean.rows,mean.cols,mean.type(),channel_mean);
}

void BrandClassifier::warpper()
{
    Blob<float>* b=net->input_blobs()[0];
    float* data=b->mutable_cpu_data();
    std::cout<<this->input.size()<<std::endl;
    for(int i=0;i<b->channels();i++)
    {
        Mat temp=Mat(b->height(),b->width(),CV_32FC1,data);
        this->input.push_back(temp);
        data+=b->height()*b->width();
    }
}
int BrandClassifier::loadImage(string img_file)
{
    warpper();
    Blob<float>* b=net->input_blobs()[0];
    Mat test_image=imread(img_file);
    Mat sample;
    cv::resize(test_image,sample,{b->height(),b->width()});
    Mat sample_float;
    sample.convertTo(sample_float,CV_32FC3);
    Mat sample_normalized;
    cv::subtract(sample_float,mean_img,sample_normalized);
    cv::split(sample_normalized,input.data());
    return 0;
}
int BrandClassifier::predict()
{
    net->Forward();
    input.clear();
    return 0;
}

int BrandClassifier::analyzeResult(float* max)
{
    Blob<float>* b=net->output_blobs()[0];
    const float* data=b->cpu_data();
    auto it=max_element(data,data+b->channels());
    *max=*it;
    return (it-data);
}