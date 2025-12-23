#include "deepTracker.h"
#include <limits>

// 计算两个矩形框的IOU
static float IOU(const cv::Rect &srcRect, const cv::Rect &dstRect)
{
    cv::Rect intersection;
    intersection = srcRect & dstRect;

    auto  area_src = static_cast<float>(srcRect.area());
    auto  area_dst = static_cast<float>(dstRect.area());
    auto  area_intersection = static_cast<float>(intersection.area());
    float iou = area_intersection / (area_src + area_dst - area_intersection);
    return iou;
}

DeepTracker::DeepTracker(const std::string    &engine_name,
                         const TRACKER_CONFIG &trackerConfig)
{
    trackerConfig_ = trackerConfig;
    list_capacity_ = trackerConfig_.targetManagement
                         .maxTrackAge; // 设置默认容量为最大跟踪年龄
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
    list_ = nullptr;
    list_size_ = 0;
    // 来自配置的开关与阈值（由 parseConfigFile 填充默认值）
    enableTrackCenterStable_ = trackerConfig_.enableTrackCenterStable; // 是否启用跟踪中心稳定判断
    trackCenterStablePixelThreshold_ = trackerConfig_.trackCenterStablePixelThreshold; // 像素阈值
}

DeepTracker::~DeepTracker()
{
    if (list_ != nullptr)
    {
        delete[] list_;
        list_ = nullptr;
    }
}

TrackInfo DeepTracker::update(const cv::Mat             &img,
                              const NvMOTObjToTrackList *detectObjList,
                              const uint32_t             frameNum,
                              uint32_t                  *matchedDetectId)
{
    frameNum_ = frameNum;
    bool is_good_track = false;

    if (matchedDetectId != nullptr)
    {
        *matchedDetectId = std::numeric_limits<uint32_t>::max();
    }

    // 输出的结果存放在bbox中
    if (is_tracked_ == false)
    {
        if (img.empty() || detectObjList == nullptr ||
            detectObjList->numFilled == 0)
        {
            memset(&trackInfo_, 0, sizeof(trackInfo_));
            return trackInfo_;
        }

        // 遍历检测到的目标
        // 当前画面的中心坐标
        auto image_cx = img.cols / 2.f;
        auto image_cy = img.rows / 2.f;

        NvMOTObjToTrack *closest_class1 = nullptr; // 记录最近的目标
        NvMOTObjToTrack *closest_any = nullptr; // 记录最近的目标（不限classId）
        float            min_distance_class1 = FLT_MAX; // classId=1的最小距离
        float            min_distance_any = FLT_MAX;    // 所有目标的最小距离
        for (uint32_t numObjects = 0; numObjects < detectObjList->numFilled;
             numObjects++)
        {
            // 直接使用数组元素的指针，避免局部变量作用域问题
            NvMOTObjToTrack *pObj = &detectObjList->list[numObjects];

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
        objectToTrack_ =
            (closest_class1 != nullptr) ? closest_class1 : closest_any;

        trackInfo_.bbox.box.x0 = objectToTrack_->bbox.x;
        trackInfo_.bbox.box.x1 =
            objectToTrack_->bbox.x + objectToTrack_->bbox.width;
        trackInfo_.bbox.box.y0 = objectToTrack_->bbox.y;
        trackInfo_.bbox.box.y1 =
            objectToTrack_->bbox.y + objectToTrack_->bbox.height;
        trackInfo_.bbox.box.w = objectToTrack_->bbox.width;
        trackInfo_.bbox.box.h = objectToTrack_->bbox.height;
        trackInfo_.bbox.box.cx =
            (trackInfo_.bbox.box.x0 + trackInfo_.bbox.box.x1) / 2.f;
        trackInfo_.bbox.box.cy =
            (trackInfo_.bbox.box.y0 + trackInfo_.bbox.box.y1) / 2.f;
        trackInfo_.bbox.score = objectToTrack_->confidence;
        trackInfo_.bbox.class_id = objectToTrack_->classId;
        // mixformer_->init(img, trackInfo_.bbox);
        trackerPtr_->init(img, trackInfo_.bbox);

        if (list_ != nullptr)
        {
            delete[] list_;
            list_ = nullptr;
        }
        list_size_ = 0;
        list_ = new NvDsTargetMiscDataFrame[trackerConfig_.targetManagement
                                                .maxTrackAge];

        is_tracked_ = true;
    }
    else
    {
        // 更新跟踪器并获取跟踪结果
        // trackInfo_.bbox = mixformer_->track(img);
        trackInfo_.bbox = trackerPtr_->track(img);
        bool is_track_match_detect = true;

        if (trackInfo_.bbox.score <= 0 || trackInfo_.bbox.box.w <= 0 ||
            trackInfo_.bbox.box.h <= 0)
        {
            // 如果跟踪结果的分数小于等于0，或者宽度或高度小于等于0，认为跟踪失败
            miss_ = trackerConfig_.targetManagement.maxMiss + 1; // 跟踪失败
        }
        else
        {
            // 如果有检测结果，和检测结果对比来查看跟踪是否正确
            float iou = 0.;
            if (age_ > trackerConfig_.targetManagement.probationAge)
            {
                if (detectObjList->numFilled != 0)
                {
                    is_track_match_detect = false;
                    // 计算和所有检测frame->objectsIn.list结果的IOU
                    cv::Rect trackRect = cv::Rect(
                        trackInfo_.bbox.box.x0, trackInfo_.bbox.box.y0,
                        trackInfo_.bbox.box.x1 - trackInfo_.bbox.box.x0,
                        trackInfo_.bbox.box.y1 - trackInfo_.bbox.box.y0);
                    for (uint i = 0; i < detectObjList->numFilled; i++)
                    {
                        NvMOTObjToTrack obj = detectObjList->list[i];
                        // 计算IOU
                        cv::Rect detectionRect =
                            cv::Rect(obj.bbox.x, obj.bbox.y, obj.bbox.width,
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
                if (detectObjList->numFilled != 0)
                {
                    // 计算和所有检测frame->objectsIn.list结果的IOU
                    cv::Rect trackRect = cv::Rect(
                        trackInfo_.bbox.box.x0, trackInfo_.bbox.box.y0,
                        trackInfo_.bbox.box.x1 - trackInfo_.bbox.box.x0,
                        trackInfo_.bbox.box.y1 - trackInfo_.bbox.box.y0);
                    for (uint i = 0; i < detectObjList->numFilled; i++)
                    {
                        NvMOTObjToTrack obj = detectObjList->list[i];
                        // 计算IOU
                        cv::Rect detectionRect =
                            cv::Rect(obj.bbox.x, obj.bbox.y, obj.bbox.width,
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
            if (trackId_++ > 0xFFFFFFFF)
            {
                trackId_ = 0;
            }

            miss_ = 0;
            memset(&trackInfo_, 0, sizeof(trackInfo_));
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
        if (age_ < list_capacity_)
        {
            list_[age_].age = age_;
            list_[age_].tBbox.left = trackInfo_.bbox.box.x0;
            list_[age_].tBbox.top = trackInfo_.bbox.box.y0;
            list_[age_].tBbox.width =
                trackInfo_.bbox.box.x1 - trackInfo_.bbox.box.x0;
            list_[age_].tBbox.height =
                trackInfo_.bbox.box.y1 - trackInfo_.bbox.box.y0;
            list_[age_].confidence = trackInfo_.bbox.score;
            list_[age_].trackerState = ACTIVE;
            list_[age_].visibility = 1.0;
            list_[age_].frameNum = frameNum_;
            list_size_ = age_ + 1;
        }
        else
        {
            // 过期的跟踪数据移除，更新为新的
            for (uint32_t i = 0; i < list_capacity_ - 1; i++)
            {
                list_[i].age = list_[i + 1].age;
                list_[i].tBbox.left = list_[i + 1].tBbox.left;
                list_[i].tBbox.top = list_[i + 1].tBbox.top;
                list_[i].tBbox.width = list_[i + 1].tBbox.width;
                list_[i].tBbox.height = list_[i + 1].tBbox.height;
                list_[i].confidence = list_[i + 1].confidence;
                list_[i].trackerState = ACTIVE;
                list_[i].visibility = 1.0;
                list_[i].frameNum = frameNum_;
            }
            list_[list_capacity_ - 1].age =
                trackerConfig_.targetManagement.maxTrackAge;
            list_[list_capacity_ - 1].tBbox.left = trackInfo_.bbox.box.x0;
            list_[list_capacity_ - 1].tBbox.top = trackInfo_.bbox.box.y0;
            list_[list_capacity_ - 1].tBbox.width =
                trackInfo_.bbox.box.x1 - trackInfo_.bbox.box.x0;
            list_[list_capacity_ - 1].tBbox.height =
                trackInfo_.bbox.box.y1 - trackInfo_.bbox.box.y0;
            list_[list_capacity_ - 1].confidence = trackInfo_.bbox.score;
            list_[list_capacity_ - 1].trackerState = ACTIVE;
            list_[list_capacity_ - 1].visibility = 1.0;
            list_[list_capacity_ - 1].frameNum = frameNum_;
            list_size_ = list_capacity_;
        }

        // 计算历史跟踪目标的位置，如果跟踪目标的中心位置变化不大，且不再画面中心，认为是虚假跟踪
        if (enableTrackCenterStable_)
        {
            if (list_size_ == list_capacity_)
            {
                // 计算跟踪中心位置的平均值
                float avg_cx = 0.0f;
                float avg_cy = 0.0f;
                for (uint32_t i = 0; i < list_size_; i++)
                {
                    avg_cx +=
                        (list_[i].tBbox.left + list_[i].tBbox.width * 0.5f);
                    avg_cy +=
                        (list_[i].tBbox.top + list_[i].tBbox.height * 0.5f);
                }
                avg_cx /= list_size_;
                avg_cy /= list_size_;

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
                    for (uint32_t i = 0; i < list_size_; i++)
                    {
                        std_dev_cx += std::pow((list_[i].tBbox.left +
                                                list_[i].tBbox.width * 0.5f) -
                                                   avg_cx,
                                               2);
                        std_dev_cy += std::pow((list_[i].tBbox.top +
                                                list_[i].tBbox.height * 0.5f) -
                                                   avg_cy,
                                               2);
                    }
                    std_dev_cx = std::sqrt(std_dev_cx / list_size_);
                    std_dev_cy = std::sqrt(std_dev_cy / list_size_);

                    if (std_dev_cx < trackCenterStablePixelThreshold_ &&
                        std_dev_cy < trackCenterStablePixelThreshold_)
                    {
                        // 如果平均中心位置变化很小，并且稳定在非画面中心位置，认为i是虚假跟踪
                        is_tracked_ = false;
                        age_ = 0;
                        if (trackId_++ > 0xFFFFFFFF)
                        {
                            trackId_ = 0;
                        }
                        miss_ = 0;
                        memset(&trackInfo_, 0, sizeof(trackInfo_));
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

void DeepTracker::updatePastFrameObjBatch(
    NvDsTargetMiscDataBatch *pastFrameObjBatch)
{
    if (pastFrameObjBatch != nullptr)
    {
        // pastFrameObjBatch->list，一个流对应一个list
        if (pastFrameObjBatch->list != nullptr)
        {
            // 目前只用一个流所以pastFrameObjBatch_->list[0]
            // pastFrameObjBatch->list[0].list表示一个流对应的所有目标，默认分配了512个目标内存
            if (pastFrameObjBatch->list[0].list != nullptr)
            {
                // pastFrameObjBatch->list[0].list[0].list表示一个跟踪目标的历史帧数据
                pastFrameObjBatch->list[0].list[0].list = list_;
                pastFrameObjBatch->list[0].list[0].numObj = list_size_;
                pastFrameObjBatch->list[0].list[0].classId =
                    objectToTrack_->classId;
                pastFrameObjBatch->list[0].list[0].uniqueId = trackId_;
                pastFrameObjBatch->list[0].list[0].numAllocated =
                    trackerConfig_.targetManagement.maxTrackAge;
            }

            pastFrameObjBatch->list[0].numFilled = 1;
            pastFrameObjBatch->list[0].streamID = 0;
            pastFrameObjBatch->list[0].surfaceStreamID = 0;
        }
        pastFrameObjBatch->numFilled = 1;
    }
}
