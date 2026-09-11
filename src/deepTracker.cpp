#include "deepTracker.h"
#include <algorithm>
#include <cmath>
#include <limits>

// 计算两个矩形框的IOU
static float IOU(const cv::Rect2f &srcRect, const cv::Rect2f &dstRect)
{
    cv::Rect2f intersection;
    intersection = srcRect & dstRect;

    auto  area_src = static_cast<float>(srcRect.area());
    auto  area_dst = static_cast<float>(dstRect.area());
    auto  area_intersection = static_cast<float>(intersection.area());
    const float union_area = area_src + area_dst - area_intersection;
    return union_area > 0.0f ? area_intersection / union_area : 0.0f;
}

DeepTracker::DeepTracker(const std::string    &engine_name,
                         const TRACKER_CONFIG &trackerConfig)
{
    trackerConfig_ = trackerConfig;
    list_capacity_ = std::max<uint32_t>(
        1, trackerConfig_.targetManagement.maxTrackAge);
    is_tracked_ = false;
    confirmAgeThreshold_ = trackerConfig_.confirmAgeThreshold;
    age_ = 0;
    trackId_ = 0;
    miss_ = 0;
    if (trackerConfig_.modelName == MODEL_SUTRACK)
    {
        trackerPtr_ = std::make_unique<SuTrackTRT>(engine_name);
    }
    else if (trackerConfig_.modelName == MODEL_OSTRACK)
    {
        trackerPtr_ = std::make_unique<OstrackTRT>(engine_name);
    }
    else if (trackerConfig_.modelName == MODEL_MIXFORMERV2)
    {
        auto mixformerPtr = std::make_unique<MixformerV2TRT>(engine_name);
        mixformerPtr->setUpdateInterval(
            trackerConfig_.mixformerV2.updateInterval);
        mixformerPtr->setMaxScoreDecay(
            trackerConfig_.mixformerV2.maxScoreDecay);
        mixformerPtr->setTemplateUpdateScoreThreshold(
            trackerConfig_.mixformerV2.templateUpdateScoreThreshold);
        mixformerPtr->setTemplateFactor(
            trackerConfig_.mixformerV2.templateFactor);
        mixformerPtr->setSearchFactor(
            trackerConfig_.mixformerV2.searchFactor);
        trackerPtr_ = std::move(mixformerPtr);
    }
    else if (trackerConfig_.modelName == MODEL_NANOTRACK)
    {
        std::unique_ptr<NanotrackTRT> nanotrackPtr;
        if (trackerConfig_.nanotrack.mode == NANOTRACK_MODE_MERGE)
        {
            std::string merge_engine = trackerConfig_.nanotrack.mergeEngine;
            if (merge_engine.empty())
            {
                merge_engine = engine_name;
            }
            if (merge_engine.empty())
            {
                std::cerr << "Nanotrack merge engine path is empty." << std::endl;
            }
            nanotrackPtr = std::make_unique<NanotrackTRT>(merge_engine);
        }
        else
        {
            if (trackerConfig_.nanotrack.headEngine.empty() ||
                trackerConfig_.nanotrack.backboneEngine.empty())
            {
                std::cerr << "Nanotrack head/backbone engine path is empty." << std::endl;
            }
            nanotrackPtr = std::make_unique<NanotrackTRT>(
                trackerConfig_.nanotrack.headEngine,
                trackerConfig_.nanotrack.backboneEngine,
                trackerConfig_.nanotrack.searchBackboneEngine);
        }

        if (trackerConfig_.nanotrack.exemplarSize > 0)
        {
            nanotrackPtr->setExemplarSize(trackerConfig_.nanotrack.exemplarSize);
        }
        if (trackerConfig_.nanotrack.instanceSize > 0)
        {
            nanotrackPtr->setInstanceSize(trackerConfig_.nanotrack.instanceSize);
        }
        trackerPtr_ = std::move(nanotrackPtr);
    }
    else
    {
        std::cerr << "Unsupported model name: " << trackerConfig_.modelName
                  << std::endl;
    }

    frameNum_ = 0;
    // 来自配置的开关与阈值（由 parseConfigFile 填充默认值）
    enableTrackCenterStable_ = trackerConfig_.enableTrackCenterStable; // 是否启用跟踪中心稳定判断
    trackCenterStablePixelThreshold_ = trackerConfig_.trackCenterStablePixelThreshold; // 像素阈值
}

DeepTracker::~DeepTracker() = default;

TrackInfo DeepTracker::update(const cv::Mat             &img,
                              const NvMOTObjToTrackList *detectObjList,
                              const uint32_t             frameNum,
                              uint32_t                  *matchedDetectId)
{
    frameNum_ = frameNum;
    bool is_good_track = false;
    trackInfo_.initialized = false;
    trackInfo_.reset = false;

    if (matchedDetectId != nullptr)
    {
        *matchedDetectId = std::numeric_limits<uint32_t>::max();
    }

    // 输出的结果存放在bbox中
    if (is_tracked_ == false)
    {
        if (img.empty() || detectObjList == nullptr ||
            detectObjList->list == nullptr || detectObjList->numFilled == 0)
        {
            memset(&trackInfo_, 0, sizeof(trackInfo_));
            return trackInfo_;
        }

        // 遍历检测到的目标
        // 当前画面的中心坐标
        auto image_cx = img.cols / 2.f;
        auto image_cy = img.rows / 2.f;

        const NvMOTObjToTrack *closest_class1 = nullptr; // 记录最近的目标
        const NvMOTObjToTrack *closest_any = nullptr; // 记录最近的目标（不限classId）
        float            min_distance_class1 = FLT_MAX; // classId=1的最小距离
        float            min_distance_any = FLT_MAX;    // 所有目标的最小距离
        for (uint32_t numObjects = 0; numObjects < detectObjList->numFilled;
             numObjects++)
        {
            // 直接使用数组元素的指针，避免局部变量作用域问题
            const NvMOTObjToTrack *pObj = &detectObjList->list[numObjects];

            // 计算目标中心到图像中心的曼哈顿距离
            float obj_cx = pObj->bbox.x + pObj->bbox.width * 0.5f;
            float obj_cy = pObj->bbox.y + pObj->bbox.height * 0.5f;
            float distance =
                std::abs(obj_cx - image_cx) + std::abs(obj_cy - image_cy);

            // 更新所有目标中的最近距离
            if (distance < min_distance_any)
            {
                min_distance_any = distance;
                closest_any = pObj;
            }

            // 如果是classId=1的目标，更新classId=1的最近距离
            if (pObj->classId == 1 && distance < min_distance_class1)
            {
                min_distance_class1 = distance;
                closest_class1 = pObj;
            }
        }
        // 优先选择classId=1的目标，若无则选最近目标
        const NvMOTObjToTrack *objectToTrack =
            (closest_class1 != nullptr) ? closest_class1 : closest_any;

        if (objectToTrack == nullptr || trackerPtr_ == nullptr ||
            !std::isfinite(objectToTrack->bbox.x) ||
            !std::isfinite(objectToTrack->bbox.y) ||
            !std::isfinite(objectToTrack->bbox.width) ||
            !std::isfinite(objectToTrack->bbox.height) ||
            objectToTrack->bbox.width <= 0 || objectToTrack->bbox.height <= 0)
        {
            memset(&trackInfo_, 0, sizeof(trackInfo_));
            return trackInfo_;
        }

        trackInfo_.bbox.box.x0 = objectToTrack->bbox.x;
        trackInfo_.bbox.box.x1 =
            objectToTrack->bbox.x + objectToTrack->bbox.width;
        trackInfo_.bbox.box.y0 = objectToTrack->bbox.y;
        trackInfo_.bbox.box.y1 =
            objectToTrack->bbox.y + objectToTrack->bbox.height;
        trackInfo_.bbox.box.w = objectToTrack->bbox.width;
        trackInfo_.bbox.box.h = objectToTrack->bbox.height;
        trackInfo_.bbox.box.cx =
            (trackInfo_.bbox.box.x0 + trackInfo_.bbox.box.x1) / 2.f;
        trackInfo_.bbox.box.cy =
            (trackInfo_.bbox.box.y0 + trackInfo_.bbox.box.y1) / 2.f;
        trackInfo_.bbox.score = objectToTrack->confidence;
        trackInfo_.bbox.class_id = objectToTrack->classId;
        if (trackerPtr_->initRgba(img, trackInfo_.bbox) != 0)
        {
            std::cerr << "SOT tracker initialization failed" << std::endl;
            history_.clear();
            memset(&trackInfo_, 0, sizeof(trackInfo_));
            return trackInfo_;
        }
        history_.clear();
        is_tracked_ = true;
        trackInfo_.initialized = true;
    }
    else
    {
        // 更新跟踪器并获取跟踪结果
        // trackInfo_.bbox = mixformer_->track(img);
        if (trackerPtr_ == nullptr)
        {
            is_tracked_ = false;
            memset(&trackInfo_, 0, sizeof(trackInfo_));
            return trackInfo_;
        }
        trackInfo_.bbox = trackerPtr_->trackRgba(img);
        bool is_track_match_detect = true;

        if (!std::isfinite(trackInfo_.bbox.score) ||
            !std::isfinite(trackInfo_.bbox.box.x0) ||
            !std::isfinite(trackInfo_.bbox.box.y0) ||
            !std::isfinite(trackInfo_.bbox.box.w) ||
            !std::isfinite(trackInfo_.bbox.box.h) ||
            trackInfo_.bbox.score <= 0 || trackInfo_.bbox.box.w <= 0 ||
            trackInfo_.bbox.box.h <= 0)
        {
            // 如果跟踪结果的分数小于等于0，或者宽度或高度小于等于0，认为跟踪失败
            miss_ = trackerConfig_.targetManagement.maxMiss + 1; // 跟踪失败
        }
        else
        {
            // 如果有检测结果，和检测结果对比来查看跟踪是否正确
            float iou = 0.;
            const bool hasDetections = detectObjList != nullptr &&
                                       detectObjList->list != nullptr &&
                                       detectObjList->numFilled != 0;
            if (age_ > trackerConfig_.targetManagement.probationAge)
            {
                if (hasDetections)
                {
                    is_track_match_detect = false;
                    // 计算和所有检测frame->objectsIn.list结果的IOU
                    cv::Rect2f trackRect(
                        trackInfo_.bbox.box.x0, trackInfo_.bbox.box.y0,
                        trackInfo_.bbox.box.x1 - trackInfo_.bbox.box.x0,
                        trackInfo_.bbox.box.y1 - trackInfo_.bbox.box.y0);
                    for (uint i = 0; i < detectObjList->numFilled; i++)
                    {
                        NvMOTObjToTrack obj = detectObjList->list[i];
                        // 计算IOU
                        if (!std::isfinite(obj.bbox.x) ||
                            !std::isfinite(obj.bbox.y) ||
                            !std::isfinite(obj.bbox.width) ||
                            !std::isfinite(obj.bbox.height) ||
                            obj.bbox.width <= 0 || obj.bbox.height <= 0)
                            continue;
                        cv::Rect2f detectionRect(
                            obj.bbox.x, obj.bbox.y, obj.bbox.width,
                            obj.bbox.height);

                        iou = IOU(detectionRect, trackRect);
                        if (iou > trackerConfig_.targetManagement.iouThreshold)
                        {
                            // 如果IOU大于0.5，认为跟踪成功
                            trackInfo_.bbox.class_id = obj.classId;
                            if (matchedDetectId != nullptr)
                            {
                                *matchedDetectId = static_cast<uint32_t>(i);
                            }
                            is_track_match_detect = true;
                            break;
                        }
                    }
                }

                // 如果跟踪置信度小于阈值或者前面和检测没有匹配的
                float maxBoxWidth =
                    img.cols *
                    trackerConfig_.targetManagement.trackBoxWidthThreshold;
                float maxBoxHeight =
                    img.rows *
                    trackerConfig_.targetManagement.trackBoxHeightThreshold;
                if (trackInfo_.bbox.score <
                        trackerConfig_.targetManagement.scoreThreshold ||
                    !is_track_match_detect ||
                    trackInfo_.bbox.box.w > maxBoxWidth ||
                    trackInfo_.bbox.box.h > maxBoxHeight)
                {
                    miss_++;
                    is_good_track = false;
                }
                else
                {
                    miss_ = 0;
                    age_++;
                    is_good_track = true;
                }
            }
            else
            {
                // 如果跟踪确认次数小于阈值，当检测结果为空时，认为失败，也就说必须连续有检测结果,且检测结果和跟踪结果的IOU大于阈值，才能认为跟踪成功
                is_track_match_detect = false;
                if (hasDetections)
                {
                    // 计算和所有检测frame->objectsIn.list结果的IOU
                    cv::Rect2f trackRect(
                        trackInfo_.bbox.box.x0, trackInfo_.bbox.box.y0,
                        trackInfo_.bbox.box.x1 - trackInfo_.bbox.box.x0,
                        trackInfo_.bbox.box.y1 - trackInfo_.bbox.box.y0);
                    for (uint i = 0; i < detectObjList->numFilled; i++)
                    {
                        NvMOTObjToTrack obj = detectObjList->list[i];
                        // 计算IOU
                        if (!std::isfinite(obj.bbox.x) ||
                            !std::isfinite(obj.bbox.y) ||
                            !std::isfinite(obj.bbox.width) ||
                            !std::isfinite(obj.bbox.height) ||
                            obj.bbox.width <= 0 || obj.bbox.height <= 0)
                            continue;
                        cv::Rect2f detectionRect(
                            obj.bbox.x, obj.bbox.y, obj.bbox.width,
                            obj.bbox.height);

                        iou = IOU(detectionRect, trackRect);
                        if (iou > trackerConfig_.targetManagement.iouThreshold)
                        {
                            // 如果IOU大于0.5，认为跟踪成功
                            trackInfo_.bbox.score = obj.confidence;
                            trackInfo_.bbox.class_id = obj.classId;
                            if (matchedDetectId != nullptr)
                            {
                                *matchedDetectId = static_cast<uint32_t>(i);
                            }
                            is_track_match_detect = true;
                            break;
                        }
                    }
                }
                if (!is_track_match_detect)
                {
                    miss_ =
                        trackerConfig_.targetManagement.maxMiss + 1; // 跟踪失败
                }
                else
                {
                    miss_ = 0;
                    age_++;
                    is_good_track = true;
                }
            }
        }

        if (miss_ > trackerConfig_.targetManagement.maxMiss)
        {
            // 失败次数大于阈值，重置跟踪
            is_tracked_ = false;
            age_ = 0;
            trackId_ = (trackId_ >= 0xFFFFFFFFLL) ? 0 : trackId_ + 1;

            miss_ = 0;
            memset(&trackInfo_, 0, sizeof(trackInfo_));
            trackInfo_.reset = true;
            return trackInfo_;
        }

        if (age_ < confirmAgeThreshold_)
        {
            memset(&trackInfo_, 0, sizeof(trackInfo_));
            return trackInfo_;
        }
    }

    if (is_good_track)
    {
        NvDsTargetMiscDataFrame sample{};
        sample.age = age_;
        sample.tBbox.left = trackInfo_.bbox.box.x0;
        sample.tBbox.top = trackInfo_.bbox.box.y0;
        sample.tBbox.width = trackInfo_.bbox.box.x1 - trackInfo_.bbox.box.x0;
        sample.tBbox.height = trackInfo_.bbox.box.y1 - trackInfo_.bbox.box.y0;
        sample.confidence = trackInfo_.bbox.score;
        sample.trackerState = ACTIVE;
        sample.visibility = 1.0;
        sample.frameNum = frameNum_;
        if (history_.size() == list_capacity_)
            history_.pop_front();
        history_.push_back(sample);

        // 计算历史跟踪目标的位置，如果跟踪目标的中心位置变化不大，且不再画面中心，认为是虚假跟踪
        if (enableTrackCenterStable_)
        {
            if (history_.size() == list_capacity_)
            {
                // 计算跟踪中心位置的平均值
                float avg_cx = 0.0f;
                float avg_cy = 0.0f;
                for (const auto &entry : history_)
                {
                    avg_cx += entry.tBbox.left + entry.tBbox.width * 0.5f;
                    avg_cy += entry.tBbox.top + entry.tBbox.height * 0.5f;
                }
                avg_cx /= history_.size();
                avg_cy /= history_.size();

                // 判断平均中心位置是否位于画面中心
                bool isCentered = false;
                auto imageCenterThreshold = 100;
                if (std::abs(avg_cx - (img.cols / 2.f)) <
                        imageCenterThreshold &&
                    std::abs(avg_cy - (img.rows / 2.f)) < imageCenterThreshold)
                {
                    // 如果平均中心位置接近画面中心，认为跟踪稳定
                    isCentered = true;
                }

                // 判断平均中心位置是否变换很小
                if (isCentered == false)
                {
                    // 计算中心位置标准差
                    float std_dev_cx = 0.0f;
                    float std_dev_cy = 0.0f;
                    for (const auto &entry : history_)
                    {
                        std_dev_cx += std::pow(entry.tBbox.left +
                                                  entry.tBbox.width * 0.5f - avg_cx,
                                              2);
                        std_dev_cy += std::pow(entry.tBbox.top +
                                                  entry.tBbox.height * 0.5f - avg_cy,
                                              2);
                    }
                    std_dev_cx = std::sqrt(std_dev_cx / history_.size());
                    std_dev_cy = std::sqrt(std_dev_cy / history_.size());

                    if (std_dev_cx < trackCenterStablePixelThreshold_ &&
                        std_dev_cy < trackCenterStablePixelThreshold_)
                    {
                        // 如果平均中心位置变化很小，并且稳定在非画面中心位置，认为i是虚假跟踪
                        is_tracked_ = false;
                        age_ = 0;
                        trackId_ = (trackId_ >= 0xFFFFFFFFLL) ? 0 : trackId_ + 1;
                        miss_ = 0;
                        memset(&trackInfo_, 0, sizeof(trackInfo_));
                        trackInfo_.reset = true;
                        return trackInfo_;
                    }
                }
            }
        }
    }

    trackInfo_.age = age_;
    trackInfo_.trackId = trackId_;
    trackInfo_.miss = miss_;

    return trackInfo_;
}
