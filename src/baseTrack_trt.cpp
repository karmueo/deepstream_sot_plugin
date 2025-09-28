#include "baseTrack_trt.h"
#include <fstream>
#include <cassert>

std::vector<float> hann(int sz)
{
    std::vector<float> hann1d(sz);
    std::vector<float> hann2d(sz * sz);
    for (int i = 1; i < sz + 1; i++)
    {
        float w = 0.5f - 0.5f * std::cos(2 * 3.1415926535898f * i / (sz + 1));
        hann1d[i - 1] = w;
    }
    for (int i = 0; i < sz; i++)
    {
        for (int j = 0; j < sz; j++)
        {
            hann2d[i * sz + j] = hann1d[i] * hann1d[j];
        }
    }
    return hann2d;
}

BaseTrackTRT::BaseTrackTRT(const std::string &engine_name)
{
    // deserialize engine
    this->deserialize_engine(engine_name);
    CHECK(cudaStreamCreate(&this->stream));
}

BaseTrackTRT::~BaseTrackTRT()
{
    cudaStreamDestroy(this->stream);
    delete this->context;
    delete this->engine;
    delete this->runtime;
    delete[] this->trt_model_stream;
}

void BaseTrackTRT::deserialize_engine(const std::string &engine_name)
{
    // create a model using the API directly and serialize it to a stream
    // char *trt_model_stream{nullptr};
    size_t size{0};
    std::ifstream file(engine_name, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        this->trt_model_stream = new char[size];
        assert(this->trt_model_stream);
        file.read(trt_model_stream, size);
        file.close();
    }

    this->runtime = createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    this->engine = this->runtime->deserializeCudaEngine(trt_model_stream,
                                                        size);
    assert(this->engine != nullptr);

    this->context = this->engine->createExecutionContext();
    assert(context != nullptr);
}

int BaseTrackTRT::sample_target(const cv::Mat &im, cv::Mat &croped, DrBBox target_bb, float search_area_factor, int output_sz, float &resize_factor)
{
    if (target_bb.w <= 0 || target_bb.h <= 0 || target_bb.cx <= 0 || target_bb.cy <= 0)
    {
        std::cout << "target_bb is out of range" << std::endl;
        return -1;
    }

    int x = target_bb.x0;
    int y = target_bb.y0;
    int w = target_bb.w;
    int h = target_bb.h;
    int crop_sz = std::ceil(std::sqrt(w * h) * search_area_factor);

    float cx = target_bb.cx;
    float cy = target_bb.cy;
    int x1 = std::round(cx - crop_sz * 0.5);
    int y1 = std::round(cy - crop_sz * 0.5);

    int x2 = x1 + crop_sz;
    int y2 = y1 + crop_sz;

    int x1_pad = std::max(0, -x1);
    int x2_pad = std::max(x2 - im.cols + 1, 0);

    int y1_pad = std::max(0, -y1);
    int y2_pad = std::max(y2 - im.rows + 1, 0);

    // Crop target
    cv::Rect roi_rect(x1 + x1_pad, y1 + y1_pad, (x2 - x2_pad) - (x1 + x1_pad), (y2 - y2_pad) - (y1 + y1_pad));
    if (roi_rect.x < 0 || roi_rect.y < 0 || roi_rect.width <= 0 || roi_rect.height <= 0)
    {
        std::cout << "roi_rect is out of range" << std::endl;
        return -1;
    }
    cv::Mat roi = im(roi_rect);

    // Pad
    cv::copyMakeBorder(roi, croped, y1_pad, y2_pad, x1_pad, x2_pad, cv::BORDER_CONSTANT);

    // Resize
    cv::resize(croped, croped, cv::Size(output_sz, output_sz));

    resize_factor = output_sz * 1.f / crop_sz;

    return 0;
}

void BaseTrackTRT::half_norm(const cv::Mat &img, float *input_data)
{
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;

    cv::Mat img_cp;
    img_cp = img.clone();
    cvtColor(img_cp, img_cp, cv::COLOR_BGR2RGB);

    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < img_h; h++)
        {
            for (int w = 0; w < img_w; w++)
            {
                input_data[c * img_w * img_h + h * img_w + w] =
                    cv::saturate_cast<float>((((float)img_cp.at<cv::Vec3b>(h, w)[c]) - mean_vals[c]) * norm_vals[c]);
            }
        }
    }
}

// calculate bbox
DrBBox BaseTrackTRT::cal_bbox(const float *boxes_ptr, const float &resize_factor, const float &search_size)
{
    DrBBox pred_box = {0, 0, 0, 0, 0, 0, 0, 0};

    float cx = boxes_ptr[0];
    float cy = boxes_ptr[1];
    float w = boxes_ptr[2];
    float h = boxes_ptr[3];

    if (cx < 0 || cy < 0 || w <= 0 || h <= 0)
    {
        return pred_box;
    }

    cx = cx * search_size / resize_factor;
    cy = cy * search_size / resize_factor;
    w = w * search_size / resize_factor;
    h = h * search_size / resize_factor;

    pred_box.x0 = cx - 0.5 * w;
    pred_box.y0 = cy - 0.5 * h;
    pred_box.x1 = pred_box.x0 + w;
    pred_box.y1 = pred_box.y0 + h;
    pred_box.w = w;
    pred_box.h = h;
    pred_box.cx = cx;
    pred_box.cy = cy;

    return pred_box;
}

// calculate bbox
DrBBox BaseTrackTRT::cal_bbox(const float *score_map,
                              const float *size_map,
                              const float *offset_map,
                              const int &score_map_size,
                              const int &size_map_size,
                              const int &offset_map_size,
                              const float &resize_factor,
                              const float &search_size,
                              const std::vector<float> &window,
                              const int &feat_sz,
                              float &max_score)
{
    float max_value = window[0] * score_map[0];
    int max_idx_y = 0;
    int max_idx_x = 0;
    int max_idx = 0;
    float tmp_score = 0.f;
    for (int i = 0; i < score_map_size; i++)
    {
        tmp_score = window[i] * score_map[i];
        if (tmp_score > max_value)
        {
            max_idx = i;
            max_value = tmp_score;
        }
    }

    max_idx_y = max_idx / feat_sz;
    max_idx_x = max_idx % feat_sz;

    float cx = (max_idx_x + offset_map[max_idx_y * feat_sz + max_idx_x]) * 1.f / feat_sz;
    float cy = (max_idx_y + offset_map[feat_sz * feat_sz + max_idx_y * feat_sz + max_idx_x]) *1.f / feat_sz;
    float w = size_map[max_idx_y * feat_sz + max_idx_x];
    float h = size_map[feat_sz * feat_sz + max_idx_y * feat_sz + max_idx_x];

    cx = cx * search_size / resize_factor;
    cy = cy * search_size / resize_factor;
    w = w * search_size / resize_factor;
    h = h * search_size / resize_factor;

    DrBBox pred_box = {0, 0, 0, 0, 0, 0, 0, 0};
    pred_box.x0 = cx - 0.5 * w;
    pred_box.y0 = cy - 0.5 * h;
    pred_box.x1 = pred_box.x0 + w;
    pred_box.y1 = pred_box.y0 + h;
    pred_box.w = w;
    pred_box.h = h;
    pred_box.cx = cx;
    pred_box.cy = cy;

    max_score = max_value;
    return pred_box;
}

void BaseTrackTRT::map_box_back(DrBBox &pred_box, const float &resize_factor, const float &search_size)
{
    float cx_prev = this->state.cx;
    float cy_prev = this->state.cy;

    float half_side = 0.5 * search_size / resize_factor;

    float w = pred_box.w;
    float h = pred_box.h;
    float cx = pred_box.cx;
    float cy = pred_box.cy;

    float cx_real = cx + (cx_prev - half_side);
    float cy_real = cy + (cy_prev - half_side);

    pred_box.x0 = cx_real - 0.5 * w;
    pred_box.y0 = cy_real - 0.5 * h;
    pred_box.x1 = cx_real + 0.5 * w;
    pred_box.y1 = cy_real + 0.5 * h;
    pred_box.w = w;
    pred_box.h = h;
    pred_box.cx = cx_real;
    pred_box.cy = cy_real;
}

void BaseTrackTRT::clip_box(DrBBox &box, const int &height, const int &wight, const int &margin)
{
    box.x0 = std::min(std::max(0, int(box.x0)), wight - margin);
    box.y0 = std::min(std::max(0, int(box.y0)), height - margin);
    box.x1 = std::min(std::max(margin, int(box.x1)), wight);
    box.y1 = std::min(std::max(margin, int(box.y1)), height);
}