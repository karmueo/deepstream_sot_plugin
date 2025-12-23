#include "nanotrack_trt.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

NanotrackTRT::NanotrackTRT(const std::string &head_engine_name,
                           const std::string &backbone_engine_name,
                           const std::string &search_backbone_engine_name)
    : BaseTrackTRT(head_engine_name)
{
    this->use_merge_ = false;
    loadEngine(backbone_engine_name, this->backbone_);
    if (!search_backbone_engine_name.empty())
    {
        loadEngine(search_backbone_engine_name, this->search_backbone_);
        this->has_search_backbone_ = true;
    }

    initIOBuffer();
    ensureScoreSize(this->cfg_.score_size);
}

NanotrackTRT::NanotrackTRT(const std::string &merge_engine_name)
    : BaseTrackTRT(merge_engine_name)
{
    this->use_merge_ = true;
    initIOBuffer();
    ensureScoreSize(this->cfg_.score_size);
}

NanotrackTRT::~NanotrackTRT()
{
    destroyIOBuffer();
    if (!this->use_merge_)
    {
        releaseEngine(this->backbone_);
        if (this->has_search_backbone_)
        {
            releaseEngine(this->search_backbone_);
        }
    }
}

void NanotrackTRT::setExemplarSize(int size)
{
    if (size > 0)
    {
        this->cfg_.exemplar_size = size;
    }
}

void NanotrackTRT::setInstanceSize(int size)
{
    if (size > 0)
    {
        this->cfg_.instance_size = size;
    }
}

void NanotrackTRT::loadEngine(const std::string &engine_name,
                              TrtEngineHandle &handle)
{
    size_t size{0};
    std::ifstream file(engine_name, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        handle.model_stream = new char[size];
        assert(handle.model_stream);
        file.read(handle.model_stream, size);
        file.close();
    }

    handle.runtime = createInferRuntime(this->gLogger);
    assert(handle.runtime != nullptr);

    handle.engine =
        handle.runtime->deserializeCudaEngine(handle.model_stream, size);
    assert(handle.engine != nullptr);

    handle.context = handle.engine->createExecutionContext();
    assert(handle.context != nullptr);
}

void NanotrackTRT::releaseEngine(TrtEngineHandle &handle)
{
    delete handle.context;
    delete handle.engine;
    delete handle.runtime;
    delete[] handle.model_stream;
    handle.context = nullptr;
    handle.engine = nullptr;
    handle.runtime = nullptr;
    handle.model_stream = nullptr;
}

void NanotrackTRT::collectIONames(ICudaEngine *engine,
                                  std::vector<std::string> &inputs,
                                  std::vector<std::string> &outputs) const
{
    inputs.clear();
    outputs.clear();

    const int io_count = engine->getNbIOTensors();
    for (int i = 0; i < io_count; ++i)
    {
        const char *name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT)
        {
            inputs.emplace_back(name);
        }
        else
        {
            outputs.emplace_back(name);
        }
    }
}

int NanotrackTRT::volume(const Dims &dims) const
{
    int total = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] <= 0)
        {
            return 0;
        }
        total *= dims.d[i];
    }
    return total;
}

void NanotrackTRT::ensureScoreSize(int size)
{
    if (size > 0 && size != this->cfg_.score_size)
    {
        this->cfg_.score_size = size;
    }

    if (this->cfg_.score_size > 0)
    {
        this->window_ = build_window(this->cfg_.score_size);
        this->points_ = build_points(this->cfg_.stride, this->cfg_.score_size);
    }
}

void NanotrackTRT::initIOBuffer()
{
    if (!this->use_merge_)
    {
        std::vector<std::string> backbone_inputs;
        std::vector<std::string> backbone_outputs;
        collectIONames(this->backbone_.engine, backbone_inputs,
                       backbone_outputs);
        assert(backbone_inputs.size() == 1);
        assert(backbone_outputs.size() == 1);
        this->backbone_input_name_ = backbone_inputs[0];
        this->backbone_output_name_ = backbone_outputs[0];

        auto backbone_in_dims =
            this->backbone_.engine->getTensorShape(
                this->backbone_input_name_.c_str());
        auto backbone_out_dims =
            this->backbone_.engine->getTensorShape(
                this->backbone_output_name_.c_str());

        this->backbone_input_size_ = volume(backbone_in_dims);
        this->backbone_output_size_ = volume(backbone_out_dims);

        if (backbone_in_dims.nbDims >= 4)
        {
            this->template_input_hw_ = {
                backbone_in_dims.d[2], backbone_in_dims.d[3]};
        }

        if (this->backbone_input_size_ > 0)
        {
            CHECK(cudaMalloc(&this->dev_backbone_input_,
                             this->backbone_input_size_ * sizeof(float)));
        }
        if (this->backbone_output_size_ > 0)
        {
            CHECK(cudaMalloc(&this->dev_backbone_output_,
                             this->backbone_output_size_ * sizeof(float)));
        }

        this->backbone_output_.resize(this->backbone_output_size_);
        this->backbone_output_shape_.clear();
        for (int i = 0; i < backbone_out_dims.nbDims; ++i)
        {
            this->backbone_output_shape_.push_back(backbone_out_dims.d[i]);
        }

        if (this->has_search_backbone_)
        {
            std::vector<std::string> search_inputs;
            std::vector<std::string> search_outputs;
            collectIONames(this->search_backbone_.engine, search_inputs,
                           search_outputs);
            assert(search_inputs.size() == 1);
            assert(search_outputs.size() == 1);
            this->search_input_name_ = search_inputs[0];
            this->search_output_name_ = search_outputs[0];

            auto search_in_dims =
                this->search_backbone_.engine->getTensorShape(
                    this->search_input_name_.c_str());
            auto search_out_dims =
                this->search_backbone_.engine->getTensorShape(
                    this->search_output_name_.c_str());

            this->search_input_size_ = volume(search_in_dims);
            this->search_output_size_ = volume(search_out_dims);

            if (search_in_dims.nbDims >= 4)
            {
                this->search_input_hw_ = {
                    search_in_dims.d[2], search_in_dims.d[3]};
            }

            if (this->search_input_size_ > 0)
            {
                CHECK(cudaMalloc(&this->dev_search_input_,
                                 this->search_input_size_ * sizeof(float)));
            }
            if (this->search_output_size_ > 0)
            {
                CHECK(cudaMalloc(&this->dev_search_output_,
                                 this->search_output_size_ * sizeof(float)));
            }

            this->search_output_.resize(this->search_output_size_);
            this->search_output_shape_.clear();
            for (int i = 0; i < search_out_dims.nbDims; ++i)
            {
                this->search_output_shape_.push_back(search_out_dims.d[i]);
            }
        }
        else
        {
            this->search_input_name_ = this->backbone_input_name_;
            this->search_output_name_ = this->backbone_output_name_;
            this->search_input_size_ = this->backbone_input_size_;
            this->search_output_size_ = this->backbone_output_size_;
            this->search_input_hw_ = this->template_input_hw_;
            this->search_output_shape_ = this->backbone_output_shape_;

            if (this->search_input_size_ > 0)
            {
                CHECK(cudaMalloc(&this->dev_search_input_,
                                 this->search_input_size_ * sizeof(float)));
            }
            if (this->search_output_size_ > 0)
            {
                CHECK(cudaMalloc(&this->dev_search_output_,
                                 this->search_output_size_ * sizeof(float)));
            }

            this->search_output_.resize(this->search_output_size_);
        }

        std::vector<std::string> head_inputs;
        std::vector<std::string> head_outputs;
        collectIONames(this->engine, head_inputs, head_outputs);
        assert(head_inputs.size() == 2);
        assert(head_outputs.size() == 2);

        this->head_input_z_name_ = head_inputs[0];
        this->head_input_x_name_ = head_inputs[1];

        for (const auto &name : head_outputs)
        {
            auto dims = this->engine->getTensorShape(name.c_str());
            if (dims.nbDims >= 2 && dims.d[1] == 4)
            {
                this->head_output_loc_name_ = name;
                this->head_loc_shape_.clear();
                for (int i = 0; i < dims.nbDims; ++i)
                {
                    this->head_loc_shape_.push_back(dims.d[i]);
                }
            }
            else
            {
                this->head_output_cls_name_ = name;
                this->head_cls_shape_.clear();
                for (int i = 0; i < dims.nbDims; ++i)
                {
                    this->head_cls_shape_.push_back(dims.d[i]);
                }
            }
        }

        if ((this->head_output_cls_name_.empty() ||
             this->head_output_loc_name_.empty()) &&
            head_outputs.size() == 2)
        {
            this->head_output_cls_name_ = head_outputs[0];
            this->head_output_loc_name_ = head_outputs[1];
            auto cls_dims =
                this->engine->getTensorShape(
                    this->head_output_cls_name_.c_str());
            auto loc_dims =
                this->engine->getTensorShape(
                    this->head_output_loc_name_.c_str());
            this->head_cls_shape_.clear();
            this->head_loc_shape_.clear();
            for (int i = 0; i < cls_dims.nbDims; ++i)
            {
                this->head_cls_shape_.push_back(cls_dims.d[i]);
            }
            for (int i = 0; i < loc_dims.nbDims; ++i)
            {
                this->head_loc_shape_.push_back(loc_dims.d[i]);
            }
        }

        auto head_z_dims =
            this->engine->getTensorShape(this->head_input_z_name_.c_str());
        auto head_x_dims =
            this->engine->getTensorShape(this->head_input_x_name_.c_str());
        if (head_z_dims.nbDims >= 4)
        {
            this->head_template_hw_ = {head_z_dims.d[2], head_z_dims.d[3]};
        }
        if (head_x_dims.nbDims >= 4)
        {
            this->head_search_hw_ = {head_x_dims.d[2], head_x_dims.d[3]};
        }

        this->head_input_z_size_ = volume(head_z_dims);
        this->head_input_x_size_ = volume(head_x_dims);
        this->head_output_cls_size_ =
            volume(this->engine->getTensorShape(
                this->head_output_cls_name_.c_str()));
        this->head_output_loc_size_ =
            volume(this->engine->getTensorShape(
                this->head_output_loc_name_.c_str()));

        if (this->head_input_z_size_ > 0)
        {
            CHECK(cudaMalloc(&this->dev_head_input_z_,
                             this->head_input_z_size_ * sizeof(float)));
        }
        if (this->head_input_x_size_ > 0)
        {
            CHECK(cudaMalloc(&this->dev_head_input_x_,
                             this->head_input_x_size_ * sizeof(float)));
        }
        if (this->head_output_cls_size_ > 0)
        {
            CHECK(cudaMalloc(&this->dev_head_output_cls_,
                             this->head_output_cls_size_ * sizeof(float)));
        }
        if (this->head_output_loc_size_ > 0)
        {
            CHECK(cudaMalloc(&this->dev_head_output_loc_,
                             this->head_output_loc_size_ * sizeof(float)));
        }

        this->head_output_cls_.resize(this->head_output_cls_size_);
        this->head_output_loc_.resize(this->head_output_loc_size_);

        if (!this->head_cls_shape_.empty() && this->head_cls_shape_.size() >= 4)
        {
            const int cls_size = static_cast<int>(this->head_cls_shape_[2]);
            ensureScoreSize(cls_size);
        }
        return;
    }

    std::vector<std::string> merge_inputs;
    std::vector<std::string> merge_outputs;
    collectIONames(this->engine, merge_inputs, merge_outputs);
    assert(merge_inputs.size() == 2);
    assert(merge_outputs.size() == 2);

    auto input0_dims =
        this->engine->getTensorShape(merge_inputs[0].c_str());
    auto input1_dims =
        this->engine->getTensorShape(merge_inputs[1].c_str());

    std::pair<int, int> input0_hw{-1, -1};
    std::pair<int, int> input1_hw{-1, -1};
    if (input0_dims.nbDims >= 4)
    {
        input0_hw = {input0_dims.d[2], input0_dims.d[3]};
    }
    if (input1_dims.nbDims >= 4)
    {
        input1_hw = {input1_dims.d[2], input1_dims.d[3]};
    }

    const bool input0_is_template =
        (input0_hw.first > 0 && input1_hw.first > 0)
            ? (input0_hw.first <= input1_hw.first)
            : true;
    if (input0_is_template)
    {
        this->merge_input_z_name_ = merge_inputs[0];
        this->merge_input_x_name_ = merge_inputs[1];
        this->merge_input_z_hw_ = input0_hw;
        this->merge_input_x_hw_ = input1_hw;
    }
    else
    {
        this->merge_input_z_name_ = merge_inputs[1];
        this->merge_input_x_name_ = merge_inputs[0];
        this->merge_input_z_hw_ = input1_hw;
        this->merge_input_x_hw_ = input0_hw;
    }

    this->merge_input_z_size_ =
        volume(this->engine->getTensorShape(this->merge_input_z_name_.c_str()));
    this->merge_input_x_size_ =
        volume(this->engine->getTensorShape(this->merge_input_x_name_.c_str()));

    for (const auto &name : merge_outputs)
    {
        auto dims = this->engine->getTensorShape(name.c_str());
        if (dims.nbDims >= 2 && dims.d[1] == 4)
        {
            this->merge_output_loc_name_ = name;
            this->merge_loc_shape_.clear();
            for (int i = 0; i < dims.nbDims; ++i)
            {
                this->merge_loc_shape_.push_back(dims.d[i]);
            }
        }
        else
        {
            this->merge_output_cls_name_ = name;
            this->merge_cls_shape_.clear();
            for (int i = 0; i < dims.nbDims; ++i)
            {
                this->merge_cls_shape_.push_back(dims.d[i]);
            }
        }
    }

    if ((this->merge_output_cls_name_.empty() ||
         this->merge_output_loc_name_.empty()) &&
        merge_outputs.size() == 2)
    {
        this->merge_output_cls_name_ = merge_outputs[0];
        this->merge_output_loc_name_ = merge_outputs[1];
        auto cls_dims =
            this->engine->getTensorShape(this->merge_output_cls_name_.c_str());
        auto loc_dims =
            this->engine->getTensorShape(this->merge_output_loc_name_.c_str());
        this->merge_cls_shape_.clear();
        this->merge_loc_shape_.clear();
        for (int i = 0; i < cls_dims.nbDims; ++i)
        {
            this->merge_cls_shape_.push_back(cls_dims.d[i]);
        }
        for (int i = 0; i < loc_dims.nbDims; ++i)
        {
            this->merge_loc_shape_.push_back(loc_dims.d[i]);
        }
    }

    this->merge_output_cls_size_ =
        volume(this->engine->getTensorShape(
            this->merge_output_cls_name_.c_str()));
    this->merge_output_loc_size_ =
        volume(this->engine->getTensorShape(
            this->merge_output_loc_name_.c_str()));

    if (this->merge_input_z_size_ > 0)
    {
        CHECK(cudaMalloc(&this->dev_merge_input_z_,
                         this->merge_input_z_size_ * sizeof(float)));
    }
    if (this->merge_input_x_size_ > 0)
    {
        CHECK(cudaMalloc(&this->dev_merge_input_x_,
                         this->merge_input_x_size_ * sizeof(float)));
    }
    if (this->merge_output_cls_size_ > 0)
    {
        CHECK(cudaMalloc(&this->dev_merge_output_cls_,
                         this->merge_output_cls_size_ * sizeof(float)));
    }
    if (this->merge_output_loc_size_ > 0)
    {
        CHECK(cudaMalloc(&this->dev_merge_output_loc_,
                         this->merge_output_loc_size_ * sizeof(float)));
    }

    this->merge_output_cls_.resize(this->merge_output_cls_size_);
    this->merge_output_loc_.resize(this->merge_output_loc_size_);

    if (!this->merge_cls_shape_.empty() && this->merge_cls_shape_.size() >= 4)
    {
        const int cls_size = static_cast<int>(this->merge_cls_shape_[2]);
        ensureScoreSize(cls_size);
    }
}

void NanotrackTRT::destroyIOBuffer()
{
    if (!this->use_merge_)
    {
        if (this->dev_backbone_input_ != nullptr)
        {
            CHECK(cudaFree(this->dev_backbone_input_));
            this->dev_backbone_input_ = nullptr;
        }
        if (this->dev_backbone_output_ != nullptr)
        {
            CHECK(cudaFree(this->dev_backbone_output_));
            this->dev_backbone_output_ = nullptr;
        }
        if (this->dev_search_input_ != nullptr)
        {
            CHECK(cudaFree(this->dev_search_input_));
            this->dev_search_input_ = nullptr;
        }
        if (this->dev_search_output_ != nullptr)
        {
            CHECK(cudaFree(this->dev_search_output_));
            this->dev_search_output_ = nullptr;
        }
        if (this->dev_head_input_z_ != nullptr)
        {
            CHECK(cudaFree(this->dev_head_input_z_));
            this->dev_head_input_z_ = nullptr;
        }
        if (this->dev_head_input_x_ != nullptr)
        {
            CHECK(cudaFree(this->dev_head_input_x_));
            this->dev_head_input_x_ = nullptr;
        }
        if (this->dev_head_output_cls_ != nullptr)
        {
            CHECK(cudaFree(this->dev_head_output_cls_));
            this->dev_head_output_cls_ = nullptr;
        }
        if (this->dev_head_output_loc_ != nullptr)
        {
            CHECK(cudaFree(this->dev_head_output_loc_));
            this->dev_head_output_loc_ = nullptr;
        }
        return;
    }

    if (this->dev_merge_input_z_ != nullptr)
    {
        CHECK(cudaFree(this->dev_merge_input_z_));
        this->dev_merge_input_z_ = nullptr;
    }
    if (this->dev_merge_input_x_ != nullptr)
    {
        CHECK(cudaFree(this->dev_merge_input_x_));
        this->dev_merge_input_x_ = nullptr;
    }
    if (this->dev_merge_output_cls_ != nullptr)
    {
        CHECK(cudaFree(this->dev_merge_output_cls_));
        this->dev_merge_output_cls_ = nullptr;
    }
    if (this->dev_merge_output_loc_ != nullptr)
    {
        CHECK(cudaFree(this->dev_merge_output_loc_));
        this->dev_merge_output_loc_ = nullptr;
    }
}

int NanotrackTRT::init(const cv::Mat &img, DrOBB bbox)
{
    if (img.empty())
    {
        return -1;
    }

    bbox.box.w = bbox.box.x1 - bbox.box.x0;
    bbox.box.h = bbox.box.y1 - bbox.box.y0;
    bbox.box.cx = bbox.box.x0 + 0.5f * bbox.box.w;
    bbox.box.cy = bbox.box.y0 + 0.5f * bbox.box.h;

    this->center_pos_ = cv::Point2f(
        bbox.box.x0 + (bbox.box.w - 1.f) * 0.5f,
        bbox.box.y0 + (bbox.box.h - 1.f) * 0.5f);
    this->size_ = cv::Point2f(bbox.box.w, bbox.box.h);
    float w_z = this->size_.x +
                this->cfg_.context_amount * (this->size_.x + this->size_.y);
    float h_z = this->size_.y +
                this->cfg_.context_amount * (this->size_.x + this->size_.y);
    float s_z = std::sqrt(w_z * h_z);
    this->channel_average_ = cv::mean(img);

    if (this->use_merge_)
    {
        if (this->merge_input_z_hw_.first > 0 &&
            this->merge_input_z_hw_.first != this->cfg_.exemplar_size)
        {
            std::cerr << "Nanotrack: template size mismatch with engine input\n";
            return -1;
        }

        this->zf_ = get_subwindow(img, this->center_pos_,
                                  this->cfg_.exemplar_size,
                                  static_cast<int>(std::round(s_z)),
                                  this->channel_average_);
        this->zf_shape_ = this->subwindow_shape_;
    }
    else
    {
        auto z = get_subwindow(img, this->center_pos_, this->cfg_.exemplar_size,
                               static_cast<int>(std::round(s_z)),
                               this->channel_average_);

        if (this->template_input_hw_.first > 0 &&
            this->template_input_hw_.first != this->cfg_.exemplar_size)
        {
            std::cerr << "Nanotrack: template size mismatch with engine input\n";
            return -1;
        }

        this->zf_ = run_backbone(z, this->zf_shape_);
        this->zf_ = align_feature(this->zf_, this->zf_shape_,
                                  this->head_template_hw_, this->zf_shape_);
    }
    this->last_score_ = 1.0f;

    this->state = bbox.box;
    this->object_box.box = bbox.box;
    this->object_box.score = 1.0f;
    this->object_box.class_id = bbox.class_id;

    return 0;
}

const DrOBB &NanotrackTRT::track(const cv::Mat &img)
{
    if (img.empty() || this->zf_.empty())
    {
        std::memset(&this->object_box, 0, sizeof(DrOBB));
        return this->object_box;
    }

    float w_z = this->size_.x +
                this->cfg_.context_amount * (this->size_.x + this->size_.y);
    float h_z = this->size_.y +
                this->cfg_.context_amount * (this->size_.x + this->size_.y);
    float s_z = std::sqrt(w_z * h_z);
    float scale_z = this->cfg_.exemplar_size / s_z;
    float s_x =
        s_z *
        (static_cast<float>(this->cfg_.instance_size) /
         this->cfg_.exemplar_size);

    auto x = get_subwindow(img, this->center_pos_, this->cfg_.instance_size,
                           static_cast<int>(std::round(s_x)),
                           this->channel_average_);

    std::vector<int64_t> cls_shape;
    std::vector<int64_t> loc_shape;
    std::vector<float> cls;
    std::vector<float> loc;

    if (this->use_merge_)
    {
        if (this->merge_input_x_hw_.first > 0 &&
            this->merge_input_x_hw_.first != this->cfg_.instance_size)
        {
            std::cerr << "Nanotrack: search size mismatch with engine input\n";
            std::memset(&this->object_box, 0, sizeof(DrOBB));
            return this->object_box;
        }

        auto merge_outputs =
            run_merge(this->zf_, this->zf_shape_, x, this->subwindow_shape_,
                      cls_shape, loc_shape);
        cls = std::move(merge_outputs.first);
        loc = std::move(merge_outputs.second);
    }
    else
    {
        if (this->search_input_hw_.first > 0 &&
            this->search_input_hw_.first != this->cfg_.instance_size)
        {
            std::cerr << "Nanotrack: search size mismatch with engine input\n";
            std::memset(&this->object_box, 0, sizeof(DrOBB));
            return this->object_box;
        }

        std::vector<int64_t> xf_shape;
        auto xf = run_search_backbone(x, xf_shape);
        xf = align_feature(xf, xf_shape, this->head_search_hw_, xf_shape);

        auto head_outputs =
            run_head(this->zf_, this->zf_shape_, xf, xf_shape, cls_shape,
                     loc_shape);
        cls = std::move(head_outputs.first);
        loc = std::move(head_outputs.second);
    }

    if (cls_shape.size() >= 4)
    {
        ensureScoreSize(static_cast<int>(cls_shape[2]));
    }

    auto score = convert_score(cls, cls_shape);
    auto pred_bbox = convert_bbox(loc, loc_shape);

    if (score.empty() || pred_bbox.empty())
    {
        std::memset(&this->object_box, 0, sizeof(DrOBB));
        return this->object_box;
    }

    auto change = [](float r) { return std::max(r, 1.f / r); };
    auto sz = [](float w, float h) {
        float pad = (w + h) * 0.5f;
        return std::sqrt((w + pad) * (h + pad));
    };

    std::vector<float> penalty(score.size());
    for (size_t i = 0; i < score.size(); ++i)
    {
        float sc =
            sz(pred_bbox[2 * score.size() + i],
               pred_bbox[3 * score.size() + i]) /
            sz(this->size_.x * scale_z, this->size_.y * scale_z);
        float rc = (this->size_.x / this->size_.y) /
                   (pred_bbox[2 * score.size() + i] /
                    pred_bbox[3 * score.size() + i]);
        penalty[i] = std::exp(-(change(sc) * change(rc) - 1.f) *
                              this->cfg_.penalty_k);
    }

    std::vector<float> pscore(score.size());
    for (size_t i = 0; i < score.size(); ++i)
    {
        float s = penalty[i] * score[i];
        pscore[i] = s * (1.f - this->cfg_.window_influence) +
                    this->window_[i] * this->cfg_.window_influence;
    }

    auto best_iter = std::max_element(pscore.begin(), pscore.end());
    size_t best_idx =
        static_cast<size_t>(std::distance(pscore.begin(), best_iter));

    cv::Point2f bbox;
    bbox.x = pred_bbox[best_idx] / scale_z + this->center_pos_.x;
    bbox.y = pred_bbox[score.size() + best_idx] / scale_z + this->center_pos_.y;

    float width =
        this->size_.x * (1 - this->cfg_.lr) +
        pred_bbox[2 * score.size() + best_idx] / scale_z * this->cfg_.lr;
    float height =
        this->size_.y * (1 - this->cfg_.lr) +
        pred_bbox[3 * score.size() + best_idx] / scale_z * this->cfg_.lr;

    auto clipped =
        bbox_clip(bbox.x, bbox.y, width, height, img.rows, img.cols);
    this->center_pos_ = cv::Point2f(clipped[0], clipped[1]);
    this->size_ = cv::Point2f(clipped[2], clipped[3]);

    DrBBox out_box;
    out_box.cx = this->center_pos_.x;
    out_box.cy = this->center_pos_.y;
    out_box.w = this->size_.x;
    out_box.h = this->size_.y;
    out_box.x0 = out_box.cx - 0.5f * out_box.w;
    out_box.y0 = out_box.cy - 0.5f * out_box.h;
    out_box.x1 = out_box.x0 + out_box.w;
    out_box.y1 = out_box.y0 + out_box.h;

    this->last_score_ = score[best_idx];
    this->state = out_box;
    this->object_box.box = out_box;
    this->object_box.score = this->last_score_;

    return this->object_box;
}

std::pair<std::vector<float>, std::vector<float>> NanotrackTRT::run_merge(
    const std::vector<float> &z, const std::vector<int64_t> &z_shape,
    const std::vector<float> &x, const std::vector<int64_t> &x_shape,
    std::vector<int64_t> &cls_shape, std::vector<int64_t> &loc_shape)
{
    (void)z_shape;
    (void)x_shape;

    if (z.size() != static_cast<size_t>(this->merge_input_z_size_) ||
        x.size() != static_cast<size_t>(this->merge_input_x_size_))
    {
        cls_shape.clear();
        loc_shape.clear();
        return {};
    }

    CHECK(cudaMemcpyAsync(this->dev_merge_input_z_, z.data(),
                          this->merge_input_z_size_ * sizeof(float),
                          cudaMemcpyHostToDevice, this->stream));
    CHECK(cudaMemcpyAsync(this->dev_merge_input_x_, x.data(),
                          this->merge_input_x_size_ * sizeof(float),
                          cudaMemcpyHostToDevice, this->stream));

    this->context->setTensorAddress(this->merge_input_z_name_.c_str(),
                                    this->dev_merge_input_z_);
    this->context->setTensorAddress(this->merge_input_x_name_.c_str(),
                                    this->dev_merge_input_x_);
    this->context->setTensorAddress(this->merge_output_cls_name_.c_str(),
                                    this->dev_merge_output_cls_);
    this->context->setTensorAddress(this->merge_output_loc_name_.c_str(),
                                    this->dev_merge_output_loc_);

    this->context->enqueueV3(this->stream);

    CHECK(cudaMemcpyAsync(this->merge_output_cls_.data(),
                          this->dev_merge_output_cls_,
                          this->merge_output_cls_size_ * sizeof(float),
                          cudaMemcpyDeviceToHost, this->stream));
    CHECK(cudaMemcpyAsync(this->merge_output_loc_.data(),
                          this->dev_merge_output_loc_,
                          this->merge_output_loc_size_ * sizeof(float),
                          cudaMemcpyDeviceToHost, this->stream));

    cudaStreamSynchronize(this->stream);

    cls_shape = this->merge_cls_shape_;
    loc_shape = this->merge_loc_shape_;
    return {this->merge_output_cls_, this->merge_output_loc_};
}

std::vector<float>
NanotrackTRT::run_backbone(const std::vector<float> &input,
                           std::vector<int64_t> &out_shape)
{
    if (input.size() != static_cast<size_t>(this->backbone_input_size_))
    {
        out_shape.clear();
        return {};
    }

    CHECK(cudaMemcpyAsync(this->dev_backbone_input_, input.data(),
                          this->backbone_input_size_ * sizeof(float),
                          cudaMemcpyHostToDevice, this->stream));

    this->backbone_.context->setTensorAddress(
        this->backbone_input_name_.c_str(), this->dev_backbone_input_);
    this->backbone_.context->setTensorAddress(
        this->backbone_output_name_.c_str(), this->dev_backbone_output_);
    this->backbone_.context->enqueueV3(this->stream);

    CHECK(cudaMemcpyAsync(this->backbone_output_.data(),
                          this->dev_backbone_output_,
                          this->backbone_output_size_ * sizeof(float),
                          cudaMemcpyDeviceToHost, this->stream));

    cudaStreamSynchronize(this->stream);

    out_shape = this->backbone_output_shape_;
    return this->backbone_output_;
}

std::vector<float>
NanotrackTRT::run_search_backbone(const std::vector<float> &input,
                                  std::vector<int64_t> &out_shape)
{
    if (input.size() != static_cast<size_t>(this->search_input_size_))
    {
        out_shape.clear();
        return {};
    }

    IExecutionContext *ctx =
        this->has_search_backbone_ ? this->search_backbone_.context
                                   : this->backbone_.context;

    CHECK(cudaMemcpyAsync(this->dev_search_input_, input.data(),
                          this->search_input_size_ * sizeof(float),
                          cudaMemcpyHostToDevice, this->stream));

    ctx->setTensorAddress(this->search_input_name_.c_str(),
                          this->dev_search_input_);
    ctx->setTensorAddress(this->search_output_name_.c_str(),
                          this->dev_search_output_);
    ctx->enqueueV3(this->stream);

    CHECK(cudaMemcpyAsync(this->search_output_.data(), this->dev_search_output_,
                          this->search_output_size_ * sizeof(float),
                          cudaMemcpyDeviceToHost, this->stream));

    cudaStreamSynchronize(this->stream);

    out_shape = this->search_output_shape_;
    return this->search_output_;
}

std::pair<std::vector<float>, std::vector<float>> NanotrackTRT::run_head(
    const std::vector<float> &zf, const std::vector<int64_t> &zf_shape,
    const std::vector<float> &xf, const std::vector<int64_t> &xf_shape,
    std::vector<int64_t> &cls_shape, std::vector<int64_t> &loc_shape)
{
    (void)zf_shape;
    (void)xf_shape;

    if (zf.size() != static_cast<size_t>(this->head_input_z_size_) ||
        xf.size() != static_cast<size_t>(this->head_input_x_size_))
    {
        cls_shape.clear();
        loc_shape.clear();
        return {};
    }

    CHECK(cudaMemcpyAsync(this->dev_head_input_z_, zf.data(),
                          this->head_input_z_size_ * sizeof(float),
                          cudaMemcpyHostToDevice, this->stream));
    CHECK(cudaMemcpyAsync(this->dev_head_input_x_, xf.data(),
                          this->head_input_x_size_ * sizeof(float),
                          cudaMemcpyHostToDevice, this->stream));

    this->context->setTensorAddress(this->head_input_z_name_.c_str(),
                                    this->dev_head_input_z_);
    this->context->setTensorAddress(this->head_input_x_name_.c_str(),
                                    this->dev_head_input_x_);
    this->context->setTensorAddress(this->head_output_cls_name_.c_str(),
                                    this->dev_head_output_cls_);
    this->context->setTensorAddress(this->head_output_loc_name_.c_str(),
                                    this->dev_head_output_loc_);

    this->context->enqueueV3(this->stream);

    CHECK(cudaMemcpyAsync(this->head_output_cls_.data(),
                          this->dev_head_output_cls_,
                          this->head_output_cls_size_ * sizeof(float),
                          cudaMemcpyDeviceToHost, this->stream));
    CHECK(cudaMemcpyAsync(this->head_output_loc_.data(),
                          this->dev_head_output_loc_,
                          this->head_output_loc_size_ * sizeof(float),
                          cudaMemcpyDeviceToHost, this->stream));

    cudaStreamSynchronize(this->stream);

    cls_shape = this->head_cls_shape_;
    loc_shape = this->head_loc_shape_;
    return {this->head_output_cls_, this->head_output_loc_};
}

std::vector<float> NanotrackTRT::build_window(int size)
{
    constexpr float kPi = 3.14159265358979323846f;
    std::vector<float> hanning(size);
    for (int i = 0; i < size; ++i)
    {
        hanning[i] = 0.5f - 0.5f * std::cos(2.f * kPi * i / (size - 1));
    }
    std::vector<float> window(size * size);
    for (int y = 0; y < size; ++y)
    {
        for (int x = 0; x < size; ++x)
        {
            window[y * size + x] = hanning[y] * hanning[x];
        }
    }
    return window;
}

std::vector<cv::Point2f> NanotrackTRT::build_points(int stride, int size)
{
    std::vector<cv::Point2f> pts;
    pts.reserve(static_cast<size_t>(size * size));
    int ori = -(size / 2) * stride;
    for (int y = 0; y < size; ++y)
    {
        for (int x = 0; x < size; ++x)
        {
            pts.emplace_back(static_cast<float>(ori + stride * x),
                             static_cast<float>(ori + stride * y));
        }
    }
    return pts;
}

std::vector<float> NanotrackTRT::get_subwindow(const cv::Mat &im,
                                               const cv::Point2f &pos,
                                               int model_sz,
                                               int original_sz,
                                               const cv::Scalar &avg_chans)
{
    float c = (original_sz + 1) * 0.5f;
    int context_xmin = static_cast<int>(std::floor(pos.x - c + 0.5f));
    int context_xmax = context_xmin + original_sz - 1;
    int context_ymin = static_cast<int>(std::floor(pos.y - c + 0.5f));
    int context_ymax = context_ymin + original_sz - 1;

    int left_pad = std::max(0, -context_xmin);
    int top_pad = std::max(0, -context_ymin);
    int right_pad = std::max(0, context_xmax - im.cols + 1);
    int bottom_pad = std::max(0, context_ymax - im.rows + 1);

    cv::Mat te_im;
    if (left_pad || top_pad || right_pad || bottom_pad)
    {
        te_im = cv::Mat(im.rows + top_pad + bottom_pad,
                        im.cols + left_pad + right_pad, im.type(), avg_chans);
        im.copyTo(te_im(cv::Rect(left_pad, top_pad, im.cols, im.rows)));
    }
    else
    {
        te_im = im;
    }

    context_xmin += left_pad;
    context_xmax += left_pad;
    context_ymin += top_pad;
    context_ymax += top_pad;

    cv::Rect roi(context_xmin, context_ymin, context_xmax - context_xmin + 1,
                 context_ymax - context_ymin + 1);
    cv::Mat im_patch = te_im(roi).clone();
    if (model_sz != original_sz)
    {
        cv::resize(im_patch, im_patch, cv::Size(model_sz, model_sz));
    }

    std::vector<float> data(1 * 3 * model_sz * model_sz);
    for (int cidx = 0; cidx < 3; ++cidx)
    {
        for (int y = 0; y < model_sz; ++y)
        {
            const uint8_t *row_ptr = im_patch.ptr<uint8_t>(y);
            for (int x = 0; x < model_sz; ++x)
            {
                data[cidx * model_sz * model_sz + y * model_sz + x] =
                    static_cast<float>(row_ptr[x * 3 + cidx]);
            }
        }
    }

    this->subwindow_shape_ = {1, 3, model_sz, model_sz};
    return data;
}

std::vector<float> NanotrackTRT::align_feature(
    const std::vector<float> &feat, const std::vector<int64_t> &shape,
    const std::pair<int, int> &target_hw, std::vector<int64_t> &out_shape)
{
    if (target_hw.first <= 0 || target_hw.second <= 0 ||
        shape.size() < 4)
    {
        out_shape = shape;
        return feat;
    }

    int64_t n = shape[0], c = shape[1], h = shape[2], w = shape[3];
    int t_h = target_hw.first;
    int t_w = target_hw.second;
    if (h == t_h && w == t_w)
    {
        out_shape = shape;
        return feat;
    }
    if (h < t_h || w < t_w)
    {
        out_shape = shape;
        return feat;
    }

    int h_start = static_cast<int>((h - t_h) / 2);
    int w_start = static_cast<int>((w - t_w) / 2);
    std::vector<float> cropped(static_cast<size_t>(n * c * t_h * t_w));
    for (int64_t nc = 0; nc < n * c; ++nc)
    {
        int64_t base_in = nc * h * w;
        int64_t base_out = nc * t_h * t_w;
        for (int yy = 0; yy < t_h; ++yy)
        {
            for (int xx = 0; xx < t_w; ++xx)
            {
                cropped[base_out + yy * t_w + xx] =
                    feat[base_in + (yy + h_start) * w + (xx + w_start)];
            }
        }
    }
    out_shape = {n, c, t_h, t_w};
    return cropped;
}

std::vector<float>
NanotrackTRT::convert_score(const std::vector<float> &cls,
                            const std::vector<int64_t> &shape) const
{
    if (shape.size() < 4)
    {
        return {};
    }

    int64_t c = shape[1];
    int64_t h = shape[2];
    int64_t w = shape[3];
    int64_t hw = h * w;
    std::vector<float> score(static_cast<size_t>(hw), 0.f);
    if (c == 1)
    {
        for (int64_t i = 0; i < hw; ++i)
        {
            score[i] = 1.f / (1.f + std::exp(-cls[i]));
        }
    }
    else
    {
        for (int64_t y = 0; y < h; ++y)
        {
            for (int64_t x = 0; x < w; ++x)
            {
                int64_t idx = y * w + x;
                float s0 = cls[idx];
                float s1 = cls[hw + idx];
                float e0 = std::exp(s0);
                float e1 = std::exp(s1);
                score[idx] = e1 / (e0 + e1 + 1e-6f);
            }
        }
    }
    return score;
}

std::vector<float>
NanotrackTRT::convert_bbox(const std::vector<float> &loc,
                           const std::vector<int64_t> &shape) const
{
    if (shape.size() < 4 || this->points_.empty())
    {
        return {};
    }

    int64_t h = shape[2];
    int64_t w = shape[3];
    int64_t hw = h * w;
    if (static_cast<size_t>(hw) != this->points_.size())
    {
        return {};
    }

    std::vector<float> bbox(static_cast<size_t>(4 * hw));
    for (int64_t y = 0; y < h; ++y)
    {
        for (int64_t x = 0; x < w; ++x)
        {
            int64_t idx = y * w + x;
            float l = loc[idx];
            float t = loc[hw + idx];
            float r = loc[2 * hw + idx];
            float b = loc[3 * hw + idx];
            float x1 = this->points_[idx].x - l;
            float y1 = this->points_[idx].y - t;
            float x2 = this->points_[idx].x + r;
            float y2 = this->points_[idx].y + b;
            bbox[idx] = (x1 + x2) * 0.5f;
            bbox[hw + idx] = (y1 + y2) * 0.5f;
            bbox[2 * hw + idx] = x2 - x1;
            bbox[3 * hw + idx] = y2 - y1;
        }
    }
    return bbox;
}

std::array<float, 4>
NanotrackTRT::bbox_clip(float cx, float cy, float width, float height,
                        int rows, int cols) const
{
    cx = std::max(0.f, std::min(cx, static_cast<float>(cols)));
    cy = std::max(0.f, std::min(cy, static_cast<float>(rows)));
    width = std::max(10.f, std::min(width, static_cast<float>(cols)));
    height = std::max(10.f, std::min(height, static_cast<float>(rows)));
    return {cx, cy, width, height};
}
