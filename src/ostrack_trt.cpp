#include "ostrack_trt.h"

OstrackTRT::OstrackTRT(const std::string &engine_name) : BaseTrackTRT(engine_name)
{
    // Generate hann2d window.
    this->window = hann(this->feat_sz);

    initIOBuffer();
}

OstrackTRT::~OstrackTRT()
{
    this->destroyIOBuffer();
    delete[] this->output_score_map;
    delete[] this->output_size_map;
    delete[] this->output_offset_map;
    delete[] this->input_template;
    delete[] this->input_search;
}

void OstrackTRT::initIOBuffer()
{
    assert(this->engine->getNbIOTensors() == 5);

    auto out_dims_0 = this->engine->getTensorShape(output_0);
    for (int j = 0; j < out_dims_0.nbDims; j++)
    {
        this->output_score_map_size *= out_dims_0.d[j];
    }

    auto out_dims_1 = this->engine->getTensorShape(output_1);
    for (int j = 0; j < out_dims_1.nbDims; j++)
    {
        this->output_size_map_size *= out_dims_1.d[j];
    }

    auto out_dims_2 = this->engine->getTensorShape(output_2);
    for (int j = 0; j < out_dims_2.nbDims; j++)
    {
        this->output_offset_map_size *= out_dims_2.d[j];
    }

    this->input_template_size = 3 * this->template_size * this->template_size;
    this->input_search_size = 3 * this->search_size * this->search_size;

    // INPUT
    this->input_template = new float[input_template_size];
    this->input_search = new float[input_search_size];
    CHECK(cudaMalloc(&this->dev_input_template, this->input_template_size * sizeof(float)));
    CHECK(cudaMalloc(&this->dev_input_search, this->input_search_size * sizeof(float)));

    // OUTPUT
    this->output_score_map = new float[this->output_score_map_size];
    this->output_size_map = new float[this->output_size_map_size];
    this->output_offset_map = new float[this->output_offset_map_size];
    CHECK(cudaMalloc(&this->dev_output_score_map, this->output_score_map_size * sizeof(float)));
    CHECK(cudaMalloc(&this->dev_output_size_map, this->output_size_map_size * sizeof(float)));
    CHECK(cudaMalloc(&this->dev_output_offset_map, this->output_offset_map_size * sizeof(float)));
}

void OstrackTRT::destroyIOBuffer()
{
    CHECK(cudaFree(this->dev_input_template));
    CHECK(cudaFree(this->dev_input_search));
    CHECK(cudaFree(this->dev_output_score_map));
    CHECK(cudaFree(this->dev_output_size_map));
    CHECK(cudaFree(this->dev_output_offset_map));
}

int OstrackTRT::init(const cv::Mat &img, DrOBB bbox)
{
    cv::Mat zt_patch;
    float resize_factor = 1.f;
    bbox.box.w = bbox.box.x1 - bbox.box.x0;
    bbox.box.h = bbox.box.y1 - bbox.box.y0;
    bbox.box.cx = bbox.box.x0 + 0.5f * bbox.box.w;
    bbox.box.cy = bbox.box.y0 + 0.5f * bbox.box.h;

    int ret = sample_target(img, zt_patch, bbox.box, this->template_factor, this->template_size, resize_factor);
    if (ret != 0)
    {
        return -1;
    }

    half_norm(zt_patch, this->input_template);

    this->state = bbox.box;
    this->object_box.box = bbox.box;
    this->object_box.score = 1.0f;
    this->object_box.class_id = bbox.class_id;

    return 0;
}

const DrOBB &OstrackTRT::track(const cv::Mat &img)
{
    DrBBox pred_box;

    cv::Mat x_patch;
    float resize_factor = 1.f;
    int ret = sample_target(img, x_patch, this->state, this->search_factor, this->search_size, resize_factor);
    if (ret != 0)
    {
        memset(&this->object_box, 0, sizeof(DrOBB));
        return this->object_box;
    }

    half_norm(x_patch, this->input_search);

    // 这里已经得到模型的输入：
    // this->input_template
    // this->input_search
    // infer()之后结果保存到：
    // float *output_score_map
    // float *output_size_map
    // float *output_offset_map
    infer();

    pred_box = this->cal_bbox(this->output_score_map,
                              this->output_size_map,
                              this->output_offset_map,
                              this->output_score_map_size,
                              this->output_size_map_size,
                              this->output_offset_map_size,
                              resize_factor,
                              this->search_size,
                              this->window,
                              this->feat_sz,
                              this->object_box.score);
    this->map_box_back(pred_box, resize_factor, this->search_size);
    this->clip_box(pred_box, img.rows, img.cols, 0);

    this->state = pred_box;
    this->object_box.box = pred_box;
    return this->object_box;
}

void OstrackTRT::infer()
{
    CHECK(cudaMemcpyAsync(
        this->dev_input_template,
        this->input_template,
        this->input_template_size * sizeof(float),
        cudaMemcpyHostToDevice,
        this->stream));
    CHECK(cudaMemcpyAsync(
        this->dev_input_search,
        this->input_search,
        this->input_search_size * sizeof(float),
        cudaMemcpyHostToDevice,
        this->stream));

    context->setTensorAddress(this->input_0, this->dev_input_template);
    context->setTensorAddress(this->input_1, this->dev_input_search);
    context->setTensorAddress(this->output_0, this->dev_output_score_map);
    context->setTensorAddress(this->output_1, this->dev_output_size_map);
    context->setTensorAddress(this->output_2, this->dev_output_offset_map);

    this->context->enqueueV3(this->stream);

    CHECK(cudaMemcpyAsync(
        this->output_score_map,
        this->dev_output_score_map,
        this->output_score_map_size * sizeof(float),
        cudaMemcpyDeviceToHost,
        this->stream));
    CHECK(cudaMemcpyAsync(
        this->output_size_map,
        this->dev_output_size_map,
        this->output_size_map_size * sizeof(float),
        cudaMemcpyDeviceToHost,
        this->stream));
    CHECK(cudaMemcpyAsync(
        this->output_offset_map,
        this->dev_output_offset_map,
        this->output_offset_map_size * sizeof(float),
        cudaMemcpyDeviceToHost,
        this->stream));
    cudaStreamSynchronize(this->stream);
}