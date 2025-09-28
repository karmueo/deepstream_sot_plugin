#include "mixformerv2_trt.h"

#include <algorithm>
#include <cstring>
#include <vector>
#include <limits>

namespace
{
constexpr const char *kDefaultMixFormerEngine =
    "models/mixformerv2_online_small_fp32.engine";
}

MixformerV2TRT::MixformerV2TRT(const std::string &engine_name)
    : BaseTrackTRT(engine_name.empty() ? kDefaultMixFormerEngine : engine_name)
{
    initIOBuffer();
    this->resetMaxPredScore();
}

MixformerV2TRT::~MixformerV2TRT()
{
    destroyIOBuffer();

    delete[] this->output_pred_boxes;
    delete[] this->output_pred_scores;
    delete[] this->input_template;
    delete[] this->input_online_template;
    delete[] this->input_search;
}

void MixformerV2TRT::setTemplateSize(int size)
{
    if (size > 0)
    {
        this->template_size = size;
    }
}

void MixformerV2TRT::setSearchSize(int size)
{
    if (size > 0)
    {
        this->search_size = size;
    }
}

void MixformerV2TRT::setTemplateFactor(float factor)
{
    if (factor > 0.0f)
    {
        this->template_factor = factor;
    }
}

void MixformerV2TRT::setSearchFactor(float factor)
{
    if (factor > 0.0f)
    {
        this->search_factor = factor;
    }
}

void MixformerV2TRT::initIOBuffer()
{
    this->output_pred_boxes_size = 1;
    this->output_pred_scores_size = 1;
    this->input_template_size = 1;
    this->input_online_template_size = 1;
    this->input_search_size = 1;

    assert(this->engine->getNbIOTensors() == 5);

    auto template_dims =
        this->engine->getTensorShape(this->input_template_name);
    for (int j = 0; j < template_dims.nbDims; j++)
    {
        this->input_template_size *=
            std::max(1, static_cast<int>(template_dims.d[j]));
    }
    if (template_dims.nbDims > 2)
    {
        const int inferred_template_size =
            static_cast<int>(template_dims.d[2]);
        if (inferred_template_size > 0)
        {
            this->setTemplateSize(inferred_template_size);
        }
    }

    auto online_template_dims =
        this->engine->getTensorShape(this->input_online_template_name);
    for (int j = 0; j < online_template_dims.nbDims; j++)
    {
        this->input_online_template_size *=
            std::max(1, static_cast<int>(online_template_dims.d[j]));
    }

    auto search_dims = this->engine->getTensorShape(this->input_search_name);
    for (int j = 0; j < search_dims.nbDims; j++)
    {
        this->input_search_size *=
            std::max(1, static_cast<int>(search_dims.d[j]));
    }
    if (search_dims.nbDims > 2)
    {
        const int inferred_search_size =
            static_cast<int>(search_dims.d[2]);
        if (inferred_search_size > 0)
        {
            this->setSearchSize(inferred_search_size);
        }
    }

    auto boxes_dims = this->engine->getTensorShape(this->output_boxes_name);
    for (int j = 0; j < boxes_dims.nbDims; j++)
    {
        this->output_pred_boxes_size *=
            std::max(1, static_cast<int>(boxes_dims.d[j]));
    }

    auto scores_dims = this->engine->getTensorShape(this->output_scores_name);
    for (int j = 0; j < scores_dims.nbDims; j++)
    {
        this->output_pred_scores_size *=
            std::max(1, static_cast<int>(scores_dims.d[j]));
    }

    this->input_template = new float[this->input_template_size];
    this->input_online_template = new float[this->input_online_template_size];
    this->input_search = new float[this->input_search_size];
    CHECK(cudaMalloc(&this->dev_input_template,
                     this->input_template_size * sizeof(float)));
    CHECK(cudaMalloc(&this->dev_input_online_template,
                     this->input_online_template_size * sizeof(float)));
    CHECK(cudaMalloc(&this->dev_input_search,
                     this->input_search_size * sizeof(float)));

    this->output_pred_boxes = new float[this->output_pred_boxes_size];
    this->output_pred_scores = new float[this->output_pred_scores_size];
    CHECK(cudaMalloc(&this->dev_output_pred_boxes,
                     this->output_pred_boxes_size * sizeof(float)));
    CHECK(cudaMalloc(&this->dev_output_pred_scores,
                     this->output_pred_scores_size * sizeof(float)));
}

void MixformerV2TRT::destroyIOBuffer()
{
    if (this->dev_input_template != nullptr)
    {
        CHECK(cudaFree(this->dev_input_template));
        this->dev_input_template = nullptr;
    }
    if (this->dev_input_online_template != nullptr)
    {
        CHECK(cudaFree(this->dev_input_online_template));
        this->dev_input_online_template = nullptr;
    }
    if (this->dev_input_search != nullptr)
    {
        CHECK(cudaFree(this->dev_input_search));
        this->dev_input_search = nullptr;
    }
    if (this->dev_output_pred_boxes != nullptr)
    {
        CHECK(cudaFree(this->dev_output_pred_boxes));
        this->dev_output_pred_boxes = nullptr;
    }
    if (this->dev_output_pred_scores != nullptr)
    {
        CHECK(cudaFree(this->dev_output_pred_scores));
        this->dev_output_pred_scores = nullptr;
    }
}

int MixformerV2TRT::init(const cv::Mat &img, DrOBB bbox)
{
    cv::Mat template_patch;
    float   resize_factor = 1.f;

    bbox.box.w = bbox.box.x1 - bbox.box.x0;
    bbox.box.h = bbox.box.y1 - bbox.box.y0;
    bbox.box.cx = bbox.box.x0 + 0.5f * bbox.box.w;
    bbox.box.cy = bbox.box.y0 + 0.5f * bbox.box.h;

    int ret =
        sample_target(img, template_patch, bbox.box, this->template_factor,
                      this->template_size, resize_factor);
    if (ret != 0)
    {
        return -1;
    }

    half_norm(template_patch, this->input_template);
    std::memcpy(this->input_online_template, this->input_template,
                this->input_template_size * sizeof(float));

    this->state = bbox.box;
    this->object_box.box = bbox.box;
    this->object_box.score = 1.0f;
    this->object_box.class_id = bbox.class_id;
    this->resetMaxPredScore();
    this->frame_id = 0;

    return 0;
}

const DrOBB &MixformerV2TRT::track(const cv::Mat &img)
{
    const int interval = this->update_interval > 0 ? this->update_interval : 200;
    const int current_frame_id = this->frame_id;
    if (this->frame_id >= std::numeric_limits<int>::max())
    {
        this->frame_id = 0;
    }
    else
    {
        ++this->frame_id;
    }

    this->max_pred_score *= this->max_score_decay;

    cv::Mat search_patch;
    float   search_resize_factor = 1.f;
    int ret = sample_target(img, search_patch, this->state, this->search_factor,
                            this->search_size, search_resize_factor);
    if (ret != 0)
    {
        std::memset(&this->object_box, 0, sizeof(DrOBB));
        return this->object_box;
    }

    half_norm(search_patch, this->input_search);

    bool    has_online_template_patch = false;
    cv::Mat online_template_patch;
    float   template_resize_factor = 1.f;
    ret = sample_target(img, online_template_patch, this->state,
                        this->template_factor, this->template_size,
                        template_resize_factor);
    if (ret == 0)
    {
        has_online_template_patch = true;
    }

    infer();

    DrBBox pred_box = this->cal_bbox(this->output_pred_boxes,
                                     search_resize_factor, this->search_size);
    float  pred_score =
        this->output_pred_scores_size > 0 ? this->output_pred_scores[0] : 0.f;

    if (pred_box.w <= 0.f || pred_box.h <= 0.f)
    {
        std::memset(&this->object_box, 0, sizeof(DrOBB));
        return this->object_box;
    }

    this->map_box_back(pred_box, search_resize_factor, this->search_size);
    this->clip_box(pred_box, img.rows, img.cols, 0);

    this->state = pred_box;
    this->object_box.box = pred_box;
    this->object_box.score = pred_score;

    const bool should_update_online_template =
        (current_frame_id % interval == 0);

    const bool can_refresh_online_template =
        has_online_template_patch &&
        pred_score > this->template_update_score_threshold &&
        pred_score > this->max_pred_score;

    if (can_refresh_online_template)
    {
        if (this->new_online_template.size() !=
            static_cast<size_t>(this->input_online_template_size))
        {
            this->new_online_template.resize(this->input_online_template_size);
        }

        half_norm(online_template_patch, this->new_online_template.data());
        this->max_pred_score = pred_score;
    }

    if (should_update_online_template &&
        this->new_online_template.size() ==
            static_cast<size_t>(this->input_online_template_size))
    {
        std::memcpy(this->input_online_template,
                    this->new_online_template.data(),
                    this->input_online_template_size * sizeof(float));
    }

    return this->object_box;
}

void MixformerV2TRT::infer()
{
    CHECK(cudaMemcpyAsync(this->dev_input_template, this->input_template,
                          this->input_template_size * sizeof(float),
                          cudaMemcpyHostToDevice, this->stream));

    CHECK(cudaMemcpyAsync(this->dev_input_online_template,
                          this->input_online_template,
                          this->input_online_template_size * sizeof(float),
                          cudaMemcpyHostToDevice, this->stream));

    CHECK(cudaMemcpyAsync(this->dev_input_search, this->input_search,
                          this->input_search_size * sizeof(float),
                          cudaMemcpyHostToDevice, this->stream));

    this->context->setTensorAddress(this->input_template_name,
                                    this->dev_input_template);
    this->context->setTensorAddress(this->input_online_template_name,
                                    this->dev_input_online_template);
    this->context->setTensorAddress(this->input_search_name,
                                    this->dev_input_search);
    this->context->setTensorAddress(this->output_boxes_name,
                                    this->dev_output_pred_boxes);
    this->context->setTensorAddress(this->output_scores_name,
                                    this->dev_output_pred_scores);

    this->context->enqueueV3(this->stream);

    CHECK(cudaMemcpyAsync(this->output_pred_boxes, this->dev_output_pred_boxes,
                          this->output_pred_boxes_size * sizeof(float),
                          cudaMemcpyDeviceToHost, this->stream));

    CHECK(cudaMemcpyAsync(this->output_pred_scores,
                          this->dev_output_pred_scores,
                          this->output_pred_scores_size * sizeof(float),
                          cudaMemcpyDeviceToHost, this->stream));

    cudaStreamSynchronize(this->stream);
}
