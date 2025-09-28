#ifndef MIXFORMERV2_TRT_H
#define MIXFORMERV2_TRT_H

#include "baseTrack_trt.h"

#include <vector>

class MixformerV2TRT : public BaseTrackTRT
{
  public:
    explicit MixformerV2TRT(const std::string &engine_name);

    ~MixformerV2TRT();

    void initIOBuffer() override;

    void destroyIOBuffer() override;

    int init(const cv::Mat &img, DrOBB bbox) override;

    const DrOBB &track(const cv::Mat &img) override;

    void setUpdateInterval(int interval) { this->update_interval = interval; }

    void infer();

    void setMaxScoreDecay(float decay) { this->max_score_decay = decay; }

  void setTemplateUpdateScoreThreshold(float threshold)
  {
    this->template_update_score_threshold = threshold;
  }

    void setTemplateSize(int size);

    void setSearchSize(int size);

    void setTemplateFactor(float factor);

    void setSearchFactor(float factor);

    void resetMaxPredScore(float value = 0.f) { this->max_pred_score = value; }

    float getMaxPredScore() const { return this->max_pred_score; }

  private:
    int output_pred_boxes_size = 1;
    int output_pred_scores_size = 1;
    int input_template_size = 1;
    int input_online_template_size = 1;
    int input_search_size = 1;

    int frame_id = 0;
    int update_interval = 200;

    float *output_pred_boxes = nullptr;
    float *output_pred_scores = nullptr;
    float *input_template = nullptr;
    float *input_online_template = nullptr;
    float *input_search = nullptr;

    std::vector<float> new_online_template;

    void *dev_input_template = nullptr;
    void *dev_input_online_template = nullptr;
    void *dev_input_search = nullptr;
    void *dev_output_pred_boxes = nullptr;
    void *dev_output_pred_scores = nullptr;

    int   template_size = 112;
    int   search_size = 224;
    float template_factor = 2.0f;
    float search_factor = 4.0f;

    const char *input_template_name = "template";
    const char *input_online_template_name = "online_template";
    const char *input_search_name = "search";
    const char *output_boxes_name = "pred_boxes";
    const char *output_scores_name = "pred_scores";

    float max_pred_score = 0.f;
    float max_score_decay = 0.95f;
    float template_update_score_threshold = 0.5f;
};

#endif
