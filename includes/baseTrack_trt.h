#ifndef BASETRACK_TRT_H
#define BASETRACK_TRT_H

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <memory>
#include <assert.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <half.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "cuda_fp16.h"
#include "logging.h"

using namespace nvinfer1;

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

struct DrBBox
{
    float x0;
    float y0;
    float x1;
    float y1;
    float w;
    float h;
    float cx;
    float cy;
};

struct DrOBB
{
    DrBBox box;
    float score;
    int class_id;
};

std::vector<float> hann(int sz);

class BaseTrackTRT
{
public: 
    BaseTrackTRT(const std::string &engine_name);
    ~BaseTrackTRT();

    void deserialize_engine(const std::string &engine_name);

    virtual void initIOBuffer() = 0;

    virtual void destroyIOBuffer() = 0;

    virtual int init(const cv::Mat &img, DrOBB bbox) = 0;

    virtual const DrOBB &track(const cv::Mat &img) = 0;

protected: 
    // FIXME: 做异常处理
    int sample_target(
        const cv::Mat &im,
        cv::Mat &croped,
        DrBBox target_bb,
        float search_area_factor,
        int output_sz,
        float &resize_factor);

    void half_norm(const cv::Mat &img, float *input_data);

    DrBBox cal_bbox(const float *boxes_ptr, const float &resize_factor, const float &search_size);

    DrBBox cal_bbox(const float *score_map,
                    const float *size_map,
                    const float *offset_map,
                    const int &score_map_size,
                    const int &size_map_size,
                    const int &offset_map_size,
                    const float &resize_factor,
                    const float &search_size,
                    const std::vector<float> &window,
                    const int &feat_sz,
                    float &max_score);

    void map_box_back(DrBBox &pred_box, const float &resize_factor, const float &search_size);

    void clip_box(DrBBox &box, const int &height, const int &wight, const int &margin);

protected: 
    const float mean_vals[3] = {0.485f * 255.f,
                                0.456f * 255.f,
                                0.406f * 255.f}; // RGB
    const float norm_vals[3] = {1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f};

    char              *trt_model_stream = nullptr;
    IRuntime          *runtime          = nullptr;
    ICudaEngine       *engine           = nullptr;
    IExecutionContext *context          = nullptr;
    Logger gLogger;

    DrBBox state;
    DrOBB object_box;
    cudaStream_t stream;
};

#endif