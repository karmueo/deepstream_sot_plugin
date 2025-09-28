#include "suTrack_trt.h"
#include <cmath>
// #include <chrono>

SuTrackTRT::SuTrackTRT(const std::string &engine_name) : BaseTrackTRT(engine_name)
{
    initIOBuffer();
}

SuTrackTRT::~SuTrackTRT()
{
    this->destroyIOBuffer();
    delete[] this->output_pred_boxes;
    delete[] this->output_score;
    delete[] this->input_template;
    delete[] this->input_search;
    delete[] this->input_template_anno;
}

void SuTrackTRT::initIOBuffer()
{
    assert(this->engine->getNbIOTensors() == 5);

    auto out_dims_0 = this->engine->getTensorShape(output_0);
    for (int j = 0; j < out_dims_0.nbDims; j++)
    {
        this->output_pred_boxes_size *= out_dims_0.d[j];
    }

    auto out_dims_1 = this->engine->getTensorShape(output_1);
    for (int j = 0; j < out_dims_1.nbDims; j++)
    {
        this->output_score_size *= out_dims_1.d[j];
    }

    this->input_template_size = 3 * this->template_size * this->template_size;
    this->input_search_size = 3 * this->search_size * this->search_size;
    this->input_template_anno_size = 4;

    // INPUT
    this->input_template = new float[input_template_size];
    this->input_search = new float[input_search_size];
    this->input_template_anno = new float[input_template_anno_size];
    CHECK(cudaMalloc(&this->dev_input_template, this->input_template_size * sizeof(float)));
    CHECK(cudaMalloc(&this->dev_input_search, this->input_search_size * sizeof(float)));
    CHECK(cudaMalloc(&this->dev_input_template_anno, this->input_template_anno_size * sizeof(float)));

    // OUTPUT
    this->output_pred_boxes = new float[this->output_pred_boxes_size];
    this->output_score = new float[this->output_score_size];
    CHECK(cudaMalloc(&this->dev_output_pred_boxes, this->output_pred_boxes_size * sizeof(float)));
    CHECK(cudaMalloc(&this->dev_output_score, this->output_score_size * sizeof(float)));
}

void SuTrackTRT::destroyIOBuffer()
{
    CHECK(cudaFree(this->dev_input_template));
    CHECK(cudaFree(this->dev_input_search));
    CHECK(cudaFree(this->dev_input_template_anno));
    CHECK(cudaFree(this->dev_output_pred_boxes));
    CHECK(cudaFree(this->dev_output_score));
}

int SuTrackTRT::init(const cv::Mat &img, DrOBB bbox)
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

    DrBBox prev_box_crop = transform_image_to_crop(bbox.box, bbox.box, resize_factor, cv::Size(template_size, template_size), true);
    this->input_template_anno[0] = prev_box_crop.x0;
    this->input_template_anno[1] = prev_box_crop.y0;
    this->input_template_anno[2] = prev_box_crop.w;
    this->input_template_anno[3] = prev_box_crop.h;

    this->state = bbox.box;
    this->object_box.box = bbox.box;
    this->object_box.score = 1.0f;
    this->object_box.class_id = bbox.class_id;

    return 0;
}

const DrOBB &SuTrackTRT::track(const cv::Mat &img)
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
    // this->input_template_anno
    // infer()之后结果保存到：
    // float *output_pred_boxes
    // float *output_score
    // float *output_size_map
    // float *output_offset_map
    infer();

    pred_box = this->cal_bbox(this->output_pred_boxes, resize_factor, this->search_size);
    this->map_box_back(pred_box, resize_factor, this->search_size);
    this->clip_box(pred_box, img.rows, img.cols, 0);

    this->state = pred_box;
    this->object_box.box = pred_box;
    this->object_box.score = this->get_max_score();
    return this->object_box;
}

void SuTrackTRT::infer()
{
    // // 统计推理时间
    // auto start = std::chrono::high_resolution_clock::now();

    // DMA input batch  data to device, infer on the batch asynchronously,  and DMA output back to host
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
    CHECK(cudaMemcpyAsync(
        this->dev_input_template_anno,
        this->input_template_anno,
        this->input_template_anno_size * sizeof(float),
        cudaMemcpyHostToDevice,
        this->stream));

    context->setTensorAddress(this->input_0, this->dev_input_template);
    context->setTensorAddress(this->input_1, this->dev_input_search);
    context->setTensorAddress(this->input_2, this->dev_input_template_anno);
    context->setTensorAddress(this->output_0, this->dev_output_pred_boxes);
    context->setTensorAddress(this->output_1, this->dev_output_score);

    // inference
    this->context->enqueueV3(this->stream);

    CHECK(cudaMemcpyAsync(
        this->output_pred_boxes,
        this->dev_output_pred_boxes,
        this->output_pred_boxes_size * sizeof(float),
        cudaMemcpyDeviceToHost,
        this->stream));
    CHECK(cudaMemcpyAsync(
        this->output_score,
        this->dev_output_score,
        this->output_score_size * sizeof(float),
        cudaMemcpyDeviceToHost,
        this->stream));

    cudaStreamSynchronize(this->stream);

    auto end = std::chrono::high_resolution_clock::now();
    // double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    // printf("[SuTrackTRT::infer] 推理耗时: %.3f ms\n", duration_ms);
}

DrBBox SuTrackTRT::transform_image_to_crop(const DrBBox &box_in, const DrBBox &box_extract, float resize_factor,
                                           const cv::Size &crop_sz, bool normalize)
{
    // Calculate centers of the boxes
    float box_extract_center_x = box_extract.cx;
    float box_extract_center_y = box_extract.cy;

    float box_in_center_x = box_in.cx;
    float box_in_center_y = box_in.cy;

    // Calculate output box center
    float box_out_center_x = (crop_sz.width - 1) / 2.0f + (box_in_center_x - box_extract_center_x) * resize_factor;
    float box_out_center_y = (crop_sz.height - 1) / 2.0f + (box_in_center_y - box_extract_center_y) * resize_factor;

    // Calculate output box width and height
    float box_out_width = box_in.w * resize_factor;
    float box_out_height = box_in.h * resize_factor;

    // Create output box
    DrBBox box_out;
    box_out.x0 = box_out_center_x - 0.5f * box_out_width;
    box_out.y0 = box_out_center_y - 0.5f * box_out_height;
    box_out.x1 = box_out_center_x + 0.5f * box_out_width;
    box_out.y1 = box_out_center_y + 0.5f * box_out_height;
    box_out.w = box_out_width;
    box_out.h = box_out_height;
    box_out.cx = box_out_center_x;
    box_out.cy = box_out_center_y;

    if (normalize)
    {
        float norm_factor_width = crop_sz.width - 1;
        float norm_factor_height = crop_sz.height - 1;
        box_out.x0 /= norm_factor_width;
        box_out.y0 /= norm_factor_height;
        box_out.x1 /= norm_factor_width;
        box_out.y1 /= norm_factor_height;
        box_out.w /= norm_factor_width;
        box_out.h /= norm_factor_height;
        box_out.cx /= norm_factor_width;
        box_out.cy /= norm_factor_height;
    }

    return box_out;
}

float SuTrackTRT::get_max_score()
{
    float max_score = 0;
    // 遍历所有output_score_map_size
    for (int i = 0; i < this->output_score_size; i++)
    {
        if (this->output_score[i] > max_score)
        {
            max_score = this->output_score[i];
        }
    }
    return max_score;
}
