#ifndef OSTRACK_TRT_H
#define OSTRACK_TRT_H

#include "baseTrack_trt.h"

class OstrackTRT : public BaseTrackTRT
{
public:
    OstrackTRT(const std::string &engine_name);

    ~OstrackTRT();

    void initIOBuffer() override;

    void destroyIOBuffer() override;

    int init(const cv::Mat &img, DrOBB bbox) override;

    const DrOBB &track(const cv::Mat &img) override;

    void infer();

private:
    DrBBox transform_image_to_crop(const DrBBox &box_in, const DrBBox &box_extract, float resize_factor,
                                   const cv::Size &crop_sz, bool normalize = false);

    // 遍历计算output_score_map的最大值
    float get_max_score();

private: 
    int output_score_map_size  = 1;
    int output_size_map_size   = 1;
    int output_offset_map_size = 1;
    int input_template_size    = 1;
    int input_search_size      = 1;

    float *output_score_map  = nullptr;
    float *output_size_map   = nullptr;
    float *output_offset_map = nullptr;
    float *input_template    = nullptr;
    float *input_search      = nullptr;

    void *dev_input_template    = nullptr;
    void *dev_input_search      = nullptr;
    void *dev_output_score_map  = nullptr;
    void *dev_output_size_map   = nullptr;
    void *dev_output_offset_map = nullptr;

    int   template_size   = 192;  // 192
    int   search_size     = 384;  // 384
    float template_factor = 2.0;
    float search_factor   = 5.0;
    int   feat_sz         = 24;
    std::vector<float> window;

    const char *input_0  = "z";
    const char *input_1  = "x";
    const char *output_0 = "score_map";
    const char *output_1 = "size_map";
    const char *output_2 = "offset_map";
};

#endif