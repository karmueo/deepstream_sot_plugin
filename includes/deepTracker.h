#pragma once

#include "mixformerv2_trt.h"
#include "nvdstracker.h"
#include "suTrack_trt.h"
#include "ostrack_trt.h"


struct TrackInfo
{
    int64_t trackId;
    int64_t age;
    int64_t miss;
    DrOBB bbox;
};

enum MODEL_NAME
{
    MODEL_SUTRACK = 0,   // SUTRACK model
    MODEL_OSTRACK,       // OSTRACK model
    MODEL_MIXFORMERV2
};

struct TARGET_MANAGEMENT
{
    float    expandFactor            = 1.0f;  // 跟踪框膨胀
    uint16_t probationAge            = 5;     // 跟踪成功多少次后进入跟踪状态，默认5次
    uint16_t maxMiss                 = 10;    // 最大连续丢失次数，超过后认为跟踪失败，默认10次
    float    scoreThreshold          = 0.3f;  // 跟踪分数阈值，默认0.3
    float    iouThreshold            = 0.5f;  // IOU阈值，默认0.5
    float    trackBoxWidthThreshold  = 0.3f;  // 跟踪框宽度阈值，默认0.3
    float    trackBoxHeightThreshold = 0.3f;  // 跟踪框高度阈值，默认0.3
    uint32_t maxTrackAge             = 30;    // 最大跟踪年龄，超过后认为跟踪失败，默认30
};

struct MIXFORMERV2_CONFIG
{
    int   updateInterval = 200;                // 模板更新间隔（帧）
    float maxScoreDecay = 0.95f;               // 最大分数衰减系数
    float templateUpdateScoreThreshold = 0.5f; // 模板更新分数阈值
    int   templateSize = 112;                  // 模板尺寸
    int   searchSize = 224;                    // 搜索区域尺寸
    float templateFactor = 2.0f;               // 模板缩放系数
    float searchFactor = 4.0f;                 // 搜索区域缩放系数
};

struct TRACKER_CONFIG
{
    std::string       modelFilePath;                                   // 模型文件路径（可选，自定义）
    MODEL_NAME        modelName     = MODEL_SUTRACK;                 // 模型类型
    TARGET_MANAGEMENT targetManagement;                              // 目标管理配置
    uint32_t          confirmAgeThreshold = 5;                       // 确认跟踪的年龄阈值，默认5
    // 是否启用跟踪中心位置稳定判断（用于剔除长时间停留在图像非中心位置且晃动极小的虚假跟踪）
    bool              enableTrackCenterStable;                       // 默认开启（在解析或使用前初始化）
    // 跟踪中心位置稳定像素方差阈值（越小越严格），只有 enableTrackCenterStable 为 true 时才生效
    uint32_t          trackCenterStablePixelThreshold;               // 默认3像素（在解析或使用前初始化）
    MIXFORMERV2_CONFIG mixformerV2;                                  // MixFormerV2 特定配置
};

class DeepTracker
{

public:
    DeepTracker(const std::string &engine_name, const TRACKER_CONFIG &trackerConfig);
    ~DeepTracker();

    TrackInfo update(const cv::Mat &img,
                     const NvMOTObjToTrackList *detectObjList,
                     const uint32_t frameNum,
                     uint32_t *matchedDetectId = nullptr);

    bool isTracked() const
    {
        return is_tracked_;
    }

    TrackInfo getTrackInfo() const
    {
        return trackInfo_;
    }

    void updatePastFrameObjBatch(NvDsTargetMiscDataBatch *pastFrameObjBatch);

private:
    bool is_tracked_;
    int64_t age_;
    int64_t trackId_;
    uint32_t miss_;
    NvMOTObjToTrack *objectToTrack_;
    std::unique_ptr<BaseTrackTRT> trackerPtr_;
    TrackInfo trackInfo_;
    uint32_t frameNum_;
    NvDsTargetMiscDataFrame *list_;
    uint32_t list_capacity_ = 30;  // 默认容量为30
    uint32_t list_size_;
    bool     enableTrackCenterStable_;          // 是否启用跟踪中心位置稳定判断
    uint32_t trackCenterStablePixelThreshold_;  // 跟踪中心位置稳定判断的像素阈值，单位像素
    TRACKER_CONFIG trackerConfig_;
    uint32_t confirmAgeThreshold_;
};