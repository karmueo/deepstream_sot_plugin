#ifndef SUTRACK_TRT_H
#define SUTRACK_TRT_H

#include "baseTrack_trt.h"

class SuTrackTRT : public BaseTrackTRT
{
public:
    SuTrackTRT(const std::string &engine_name);

    ~SuTrackTRT();

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
    int output_pred_boxes_size   = 1;
    int output_score_size        = 1;
    int input_template_size      = 1;
    int input_search_size        = 1;
    int input_template_anno_size = 1;

    float *output_pred_boxes   = nullptr;
    float *output_score        = nullptr;
    float *input_template      = nullptr;
    float *input_search        = nullptr;
    float *input_template_anno = nullptr;

    void *dev_input_template      = nullptr;
    void *dev_input_search        = nullptr;
    void *dev_input_template_anno = nullptr;
    void *dev_output_pred_boxes   = nullptr;
    void *dev_output_score        = nullptr;

    int   template_size   = 112;  // 192
    int   search_size     = 224;  // 384
    float template_factor = 2.0;
    float search_factor   = 4.0;

    const char *input_0  = "template";
    const char *input_1  = "search";
    const char *input_2  = "template_anno";
    const char *output_0 = "pred_boxes";
    const char *output_1 = "score";
};

#endif