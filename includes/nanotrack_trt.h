#ifndef NANOTRACK_TRT_H
#define NANOTRACK_TRT_H

#include "baseTrack_trt.h"

#include <array>
#include <string>
#include <utility>
#include <vector>

class NanotrackTRT : public BaseTrackTRT
{
  public:
    // 构造函数，加载分段模型引擎（输入：head/backbone/search 引擎路径，输出：完成 TRT 初始化）
    NanotrackTRT(const std::string &head_engine_name,
                 const std::string &backbone_engine_name,
                 const std::string &search_backbone_engine_name = "");

    // 构造函数，加载合并模型引擎（输入：合并引擎路径，输出：完成 TRT 初始化）
    explicit NanotrackTRT(const std::string &merge_engine_name);

    ~NanotrackTRT();

    // 初始化 I/O 缓冲区（输入：无，输出：完成显存/缓存准备）
    void initIOBuffer() override;

    // 释放 I/O 缓冲区（输入：无，输出：释放显存）
    void destroyIOBuffer() override;

    // 初始化跟踪器（输入：首帧图像与初始框，输出：0 表示成功）
    int init(const cv::Mat &img, DrOBB bbox) override;

    // 执行一次跟踪（输入：当前帧图像，输出：当前帧目标框）
    const DrOBB &track(const cv::Mat &img) override;

    // 设置模板尺寸（输入：模板尺寸，输出：无）
    void setExemplarSize(int size);

    // 设置搜索尺寸（输入：搜索尺寸，输出：无）
    void setInstanceSize(int size);

    // 获取上一帧得分（输出：置信度分数）
    float getLastScore() const { return this->last_score_; }

  private:
    struct TrackerConfig
    {
        int   exemplar_size = 127;
        int   instance_size = 255;
        int   score_size = 15;
        int   stride = 16;
        float context_amount = 0.5f;
        float window_influence = 0.455f;
        float penalty_k = 0.138f;
        float lr = 0.348f;
    };

    // 收集模型输入/输出张量名（输入：engine，输出：输入/输出名列表）
    void collectIONames(ICudaEngine *engine,
                        std::vector<std::string> &inputs,
                        std::vector<std::string> &outputs) const;

    // 计算张量体积（输入：维度信息，输出：元素总数）
    int volume(const Dims &dims) const;

    // 根据 score 尺度更新窗口与点（输入：score 尺寸，输出：更新内部缓存）
    void ensureScoreSize(int size);

    // 合并模型推理（输入：模板/搜索特征，输出：分类与回归）
    std::pair<std::vector<float>, std::vector<float>>
    run_merge(const std::vector<float> &z, const std::vector<int64_t> &z_shape,
              const std::vector<float> &x, const std::vector<int64_t> &x_shape,
              std::vector<int64_t> &cls_shape, std::vector<int64_t> &loc_shape);

    // 分段模型 backbone 推理（输入：模板图像，输出：特征与形状）
    std::vector<float> run_backbone(const std::vector<float> &input,
                                    std::vector<int64_t> &out_shape);

    // 分段模型搜索 backbone 推理（输入：搜索图像，输出：特征与形状）
    std::vector<float> run_search_backbone(const std::vector<float> &input,
                                           std::vector<int64_t> &out_shape);

    // 分段模型 head 推理（输入：模板/搜索特征，输出：分类与回归）
    std::pair<std::vector<float>, std::vector<float>>
    run_head(const std::vector<float> &zf, const std::vector<int64_t> &zf_shape,
             const std::vector<float> &xf, const std::vector<int64_t> &xf_shape,
             std::vector<int64_t> &cls_shape, std::vector<int64_t> &loc_shape);

    // 构建汉宁窗（输入：score 尺寸，输出：二维窗口）
    std::vector<float> build_window(int size);

    // 构建特征点坐标（输入：步长与尺寸，输出：点列表）
    std::vector<cv::Point2f> build_points(int stride, int size);

    // 裁剪并缩放子图（输入：图像/中心/尺寸，输出：CHW 格式数据）
    std::vector<float> get_subwindow(const cv::Mat &im,
                                     const cv::Point2f &pos, int model_sz,
                                     int original_sz,
                                     const cv::Scalar &avg_chans);

    // 对齐特征图尺寸（输入：特征与目标尺寸，输出：裁剪后特征）
    std::vector<float> align_feature(const std::vector<float> &feat,
                                     const std::vector<int64_t> &shape,
                                     const std::pair<int, int> &target_hw,
                                     std::vector<int64_t> &out_shape);

    // 解析分类得分（输入：分类输出与形状，输出：得分向量）
    std::vector<float> convert_score(const std::vector<float> &cls,
                                     const std::vector<int64_t> &shape) const;

    // 解析回归框（输入：回归输出与形状，输出：bbox 向量）
    std::vector<float> convert_bbox(const std::vector<float> &loc,
                                    const std::vector<int64_t> &shape) const;

    // 边界框裁剪（输入：中心与尺寸/图像尺寸，输出：裁剪后参数）
    std::array<float, 4> bbox_clip(float cx, float cy, float width,
                                   float height, int rows, int cols) const;

  private:
    TrackerConfig cfg_;
    // 当前模式（true=合并模型，false=分段模型）
    bool use_merge_ = false;
    std::pair<int, int> head_template_hw_{-1, -1};
    std::pair<int, int> head_search_hw_{-1, -1};
    std::pair<int, int> template_input_hw_{-1, -1};
    std::pair<int, int> search_input_hw_{-1, -1};
    // 合并模型输入尺寸（模板/搜索）
    std::pair<int, int> merge_input_z_hw_{-1, -1};
    std::pair<int, int> merge_input_x_hw_{-1, -1};
    // 汉宁窗与特征点，用于后处理
    std::vector<float> window_;
    std::vector<cv::Point2f> points_;
    // 目标状态（中心与尺度）
    cv::Point2f center_pos_{0.f, 0.f};
    cv::Point2f size_{0.f, 0.f};
    cv::Scalar channel_average_;
    // 模板图像缓存（合并模型输入）
    std::vector<float> zf_;
    std::vector<int64_t> zf_shape_;
    std::vector<int64_t> subwindow_shape_;
    float last_score_ = 0.f;

    struct TrtEngineHandle
    {
        IRuntime          *runtime = nullptr;
        ICudaEngine       *engine = nullptr;
        IExecutionContext *context = nullptr;
        char              *model_stream = nullptr;
    };

    // 加载 TRT 引擎（输入：引擎路径，输出：填充 handle）
    void loadEngine(const std::string &engine_name, TrtEngineHandle &handle);

    // 释放 TRT 引擎资源（输入：handle，输出：清理资源）
    void releaseEngine(TrtEngineHandle &handle);

    TrtEngineHandle backbone_;
    TrtEngineHandle search_backbone_;
    bool has_search_backbone_ = false;

    std::string backbone_input_name_;
    std::string backbone_output_name_;
    std::string search_input_name_;
    std::string search_output_name_;
    std::string head_input_z_name_;
    std::string head_input_x_name_;
    std::string head_output_cls_name_;
    std::string head_output_loc_name_;

    int backbone_input_size_ = 1;
    int backbone_output_size_ = 1;
    int search_input_size_ = 1;
    int search_output_size_ = 1;
    int head_input_z_size_ = 1;
    int head_input_x_size_ = 1;
    int head_output_cls_size_ = 1;
    int head_output_loc_size_ = 1;

    void *dev_backbone_input_ = nullptr;
    void *dev_backbone_output_ = nullptr;
    void *dev_search_input_ = nullptr;
    void *dev_search_output_ = nullptr;
    void *dev_head_input_z_ = nullptr;
    void *dev_head_input_x_ = nullptr;
    void *dev_head_output_cls_ = nullptr;
    void *dev_head_output_loc_ = nullptr;

    std::vector<float> backbone_output_;
    std::vector<float> search_output_;
    std::vector<float> head_output_cls_;
    std::vector<float> head_output_loc_;

    std::vector<int64_t> backbone_output_shape_;
    std::vector<int64_t> search_output_shape_;
    std::vector<int64_t> head_cls_shape_;
    std::vector<int64_t> head_loc_shape_;

    // 合并模型 I/O 名称
    std::string merge_input_z_name_;
    std::string merge_input_x_name_;
    std::string merge_output_cls_name_;
    std::string merge_output_loc_name_;

    // 合并模型 I/O 尺寸
    int merge_input_z_size_ = 1;
    int merge_input_x_size_ = 1;
    int merge_output_cls_size_ = 1;
    int merge_output_loc_size_ = 1;

    // 合并模型 I/O 显存指针
    void *dev_merge_input_z_ = nullptr;
    void *dev_merge_input_x_ = nullptr;
    void *dev_merge_output_cls_ = nullptr;
    void *dev_merge_output_loc_ = nullptr;

    // 合并模型输出缓存
    std::vector<float> merge_output_cls_;
    std::vector<float> merge_output_loc_;

    // 合并模型输出形状
    std::vector<int64_t> merge_cls_shape_;
    std::vector<int64_t> merge_loc_shape_;
};

#endif
