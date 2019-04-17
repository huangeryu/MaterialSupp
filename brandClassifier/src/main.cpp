#include "BrandClassifier.h"
#include <iostream>
#include <sys/time.h>
#include <time.h>


int main(int argc,char* argv[])
{
    BrandClassifier classifier(argv[1],argv[2],argv[3]);
    string img_file;
    while(true)
    {
        std::cout<<"please enter test image patch:";
        std::cin>>img_file;
        if(!img_file.compare("quit"))break;
        cudaDeviceSynchronize();
        classifier.loadImage(img_file);
        struct timeval start,finish;
        gettimeofday(&start,NULL);
        classifier.predict();
        float probability;
        int cls=classifier.analyzeResult(&probability);
        cudaDeviceSynchronize();
        gettimeofday(&finish,NULL);
        std::cout<<"\ncls="<<cls<<",probability="<<probability<<std::endl;
        std::cout<<(finish.tv_sec-start.tv_sec)*1000.0+(finish.tv_usec-start.tv_usec)/1000.0<<"(ms)\n";
    }
}