#include "deepTracker.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
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

static float centerDistance(const NvMOTObjToTrack &obj, const DrOBB &bbox)
{
    const float obj_cx = obj.bbox.x + obj.bbox.width * 0.5f;
    const float obj_cy = obj.bbox.y + obj.bbox.height * 0.5f;
    const float dx = obj_cx - bbox.box.cx;
    const float dy = obj_cy - bbox.box.cy;
    return std::sqrt(dx * dx + dy * dy);
}

static float safeRatio(float a, float b)
{
    if (a <= 0.f || b <= 0.f)
    {
        return std::numeric_limits<float>::max();
    }
    const float ratio = a / b;
    return std::max(ratio, 1.f / ratio);
}

static bool isSotDebugEnabled()
{
    const char *env = std::getenv("SOT_DEBUG");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

DeepTracker::DeepTracker(const std::string    &engine_name,
                         const TRACKER_CONFIG &trackerConfig)
{
    trackerConfig_ = trackerConfig;
    list_capacity_ = std::max<uint32_t>(
        1, trackerConfig_.targetManagement.maxTrackAge);
    is_tracked_ = false;
    confirmAgeThreshold_ =
        std::max<uint32_t>(1, trackerConfig_.confirmAgeThreshold);
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
        if (trackerConfig_.nanotrack.windowInfluence >= 0.f)
        {
            nanotrackPtr->setWindowInfluence(
                trackerConfig_.nanotrack.windowInfluence);
        }
        if (trackerConfig_.nanotrack.penaltyK >= 0.f)
        {
            nanotrackPtr->setPenaltyK(trackerConfig_.nanotrack.penaltyK);
        }
        if (trackerConfig_.nanotrack.lr >= 0.f)
        {
            nanotrackPtr->setLearningRate(trackerConfig_.nanotrack.lr);
        }
        if (trackerConfig_.nanotrack.contextAmount >= 0.f)
        {
            nanotrackPtr->setContextAmount(
                trackerConfig_.nanotrack.contextAmount);
        }
        trackerPtr_ = std::move(nanotrackPtr);
    }
    else
    {
        std::cerr << "Unsupported model name: " << trackerConfig_.modelName
                  << std::endl;
    }

    frameNum_ = 0;
    unverifiedFrames_ = 0;
    // 来自配置的开关与阈值（由 parseConfigFile 填充默认值）
    enableTrackCenterStable_ = trackerConfig_.enableTrackCenterStable; // 是否启用跟踪中心稳定判断
    trackCenterStablePixelThreshold_ = trackerConfig_.trackCenterStablePixelThreshold; // 像素阈值
}

DeepTracker::~DeepTracker() = default;

bool DeepTracker::isTargetClass(int classId) const
{
    return trackerConfig_.targetManagement.targetClassId < 0 ||
           classId == trackerConfig_.targetManagement.targetClassId;
}

bool DeepTracker::selectTargetDetection(const NvMOTObjToTrackList *detectObjList,
                                        const cv::Mat &img,
                                        const DrOBB *referenceBox,
                                        DrOBB &selected,
                                        uint32_t *matchedDetectId) const
{
    if (matchedDetectId != nullptr)
    {
        *matchedDetectId = std::numeric_limits<uint32_t>::max();
    }
    if (detectObjList == nullptr || detectObjList->list == nullptr ||
        detectObjList->numFilled == 0)
    {
        return false;
    }

    const float refCx =
        referenceBox != nullptr ? referenceBox->box.cx : img.cols * 0.5f;
    const float refCy =
        referenceBox != nullptr ? referenceBox->box.cy : img.rows * 0.5f;
    const float norm =
        std::max(1.f, std::sqrt(static_cast<float>(img.cols * img.cols +
                                                   img.rows * img.rows)));
    float bestScore = -std::numeric_limits<float>::max();
    uint32_t bestIdx = std::numeric_limits<uint32_t>::max();

    auto selectBest = [&](bool targetOnly) {
        for (uint32_t i = 0; i < detectObjList->numFilled; ++i)
        {
            const NvMOTObjToTrack &obj = detectObjList->list[i];
            if (targetOnly && !isTargetClass(obj.classId))
            {
                continue;
            }

            const float cx = obj.bbox.x + obj.bbox.width * 0.5f;
            const float cy = obj.bbox.y + obj.bbox.height * 0.5f;
            const float dx = cx - refCx;
            const float dy = cy - refCy;
            const float distancePenalty = std::sqrt(dx * dx + dy * dy) / norm;
            const float score = obj.confidence - distancePenalty;
            if (score > bestScore)
            {
                bestScore = score;
                bestIdx = i;
            }
        }
    };

    selectBest(trackerConfig_.targetManagement.targetClassId >= 0);
    if (bestIdx == std::numeric_limits<uint32_t>::max() &&
        trackerConfig_.targetManagement.targetClassId >= 0)
    {
        if (isSotDebugEnabled())
        {
            std::cerr << "[SOT] frame " << frameNum_
                      << " no detection with targetClassId="
                      << trackerConfig_.targetManagement.targetClassId
                      << ", fallback to any class" << std::endl;
        }
        selectBest(false);
    }

    if (bestIdx == std::numeric_limits<uint32_t>::max())
    {
        return false;
    }

    const NvMOTObjToTrack &obj = detectObjList->list[bestIdx];
    selected.box.x0 = obj.bbox.x;
    selected.box.y0 = obj.bbox.y;
    selected.box.x1 = obj.bbox.x + obj.bbox.width;
    selected.box.y1 = obj.bbox.y + obj.bbox.height;
    selected.box.w = obj.bbox.width;
    selected.box.h = obj.bbox.height;
    selected.box.cx = selected.box.x0 + selected.box.w * 0.5f;
    selected.box.cy = selected.box.y0 + selected.box.h * 0.5f;
    selected.score = obj.confidence;
    selected.class_id = obj.classId;
    if (matchedDetectId != nullptr)
    {
        *matchedDetectId = bestIdx;
    }
    return true;
}

bool DeepTracker::findMatchedDetection(const NvMOTObjToTrackList *detectObjList,
                                       const DrOBB &candidate,
                                       const DrOBB &prevBox,
                                       uint32_t *matchedDetectId) const
{
    if (matchedDetectId != nullptr)
    {
        *matchedDetectId = std::numeric_limits<uint32_t>::max();
    }
    if (detectObjList == nullptr || detectObjList->list == nullptr ||
        detectObjList->numFilled == 0)
    {
        return false;
    }

    cv::Rect trackRect(candidate.box.x0, candidate.box.y0,
                       candidate.box.x1 - candidate.box.x0,
                       candidate.box.y1 - candidate.box.y0);
    const float prevDiag = std::sqrt(prevBox.box.w * prevBox.box.w +
                                     prevBox.box.h * prevBox.box.h);
    const float maxCenterDistance =
        std::max(12.f, prevDiag * trackerConfig_.targetManagement.maxCenterJumpRatio);

    float bestScore = -1.f;
    uint32_t bestIdx = std::numeric_limits<uint32_t>::max();

    auto matchBest = [&](bool targetOnly) {
        for (uint32_t i = 0; i < detectObjList->numFilled; i++)
        {
            const NvMOTObjToTrack &obj = detectObjList->list[i];
            if (targetOnly && !isTargetClass(obj.classId))
            {
                continue;
            }

            cv::Rect detectionRect(obj.bbox.x, obj.bbox.y, obj.bbox.width,
                                   obj.bbox.height);
            const float iou = IOU(detectionRect, trackRect);
            const float distance = centerDistance(obj, candidate);
            const bool matched =
                iou >= trackerConfig_.targetManagement.iouThreshold ||
                distance <= maxCenterDistance;
            if (!matched)
            {
                continue;
            }

            const float score = iou + obj.confidence;
            if (score > bestScore)
            {
                bestScore = score;
                bestIdx = i;
            }
        }
    };

    matchBest(trackerConfig_.targetManagement.targetClassId >= 0);
    if (bestIdx == std::numeric_limits<uint32_t>::max() &&
        trackerConfig_.targetManagement.targetClassId >= 0)
    {
        matchBest(false);
    }

    if (bestIdx == std::numeric_limits<uint32_t>::max())
    {
        return false;
    }
    if (matchedDetectId != nullptr)
    {
        *matchedDetectId = bestIdx;
    }
    return true;
}

bool DeepTracker::isMotionReasonable(const DrOBB &candidate,
                                     const DrOBB &prevBox,
                                     const cv::Mat &img) const
{
    if (candidate.score <= 0.f || candidate.box.w <= 0.f ||
        candidate.box.h <= 0.f)
    {
        return false;
    }

    const float maxBoxWidth =
        img.cols * trackerConfig_.targetManagement.trackBoxWidthThreshold;
    const float maxBoxHeight =
        img.rows * trackerConfig_.targetManagement.trackBoxHeightThreshold;
    if (candidate.box.w > maxBoxWidth || candidate.box.h > maxBoxHeight)
    {
        return false;
    }

    const float prevDiag = std::sqrt(prevBox.box.w * prevBox.box.w +
                                     prevBox.box.h * prevBox.box.h);
    const float dx = candidate.box.cx - prevBox.box.cx;
    const float dy = candidate.box.cy - prevBox.box.cy;
    const float distance = std::sqrt(dx * dx + dy * dy);
    if (distance >
        std::max(12.f, prevDiag * trackerConfig_.targetManagement.maxCenterJumpRatio))
    {
        return false;
    }

    const float scaleRatio = safeRatio(candidate.box.w * candidate.box.h,
                                       prevBox.box.w * prevBox.box.h);
    if (scaleRatio > trackerConfig_.targetManagement.maxScaleChangeRatio)
    {
        return false;
    }

    const float aspectRatio =
        safeRatio(candidate.box.w / candidate.box.h, prevBox.box.w / prevBox.box.h);
    return aspectRatio <= trackerConfig_.targetManagement.maxAspectChangeRatio;
}

bool DeepTracker::computeAppearanceFeature(const cv::Mat &img,
                                           const DrOBB &bbox,
                                           cv::Mat &hist,
                                           double &grayStdDev,
                                           double &edgeDensity) const
{
    if (img.empty() || bbox.box.w <= 1.f || bbox.box.h <= 1.f)
    {
        return false;
    }

    cv::Rect rect(static_cast<int>(std::round(bbox.box.x0)),
                  static_cast<int>(std::round(bbox.box.y0)),
                  static_cast<int>(std::round(bbox.box.w)),
                  static_cast<int>(std::round(bbox.box.h)));
    rect &= cv::Rect(0, 0, img.cols, img.rows);
    if (rect.width < 4 || rect.height < 4)
    {
        return false;
    }

    cv::Mat patch = img(rect).clone();
    cv::Mat hsv;
    cv::cvtColor(patch, hsv, cv::COLOR_BGR2HSV);
    int hBins = 16;
    int sBins = 16;
    int histSize[] = {hBins, sBins};
    float hRange[] = {0, 180};
    float sRange[] = {0, 256};
    const float *ranges[] = {hRange, sRange};
    int channels[] = {0, 1};
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges,
                 true, false);
    cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);

    cv::Mat gray;
    cv::cvtColor(patch, gray, cv::COLOR_BGR2GRAY);
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(gray, mean, stddev);
    grayStdDev = stddev[0];

    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);
    edgeDensity = static_cast<double>(cv::countNonZero(edges)) /
                  static_cast<double>(edges.total());
    return true;
}

bool DeepTracker::updateAppearanceReference(const cv::Mat &img,
                                            const DrOBB &bbox)
{
    return computeAppearanceFeature(img, bbox, appearanceHist_,
                                    appearanceStdDev_, appearanceEdgeDensity_);
}

bool DeepTracker::reinitializeFromDetection(const cv::Mat &img,
                                            const DrOBB &detectionBox)
{
    if (trackerPtr_ == nullptr || trackerPtr_->initRgba(img, detectionBox) != 0)
    {
        return false;
    }

    trackInfo_.bbox = detectionBox;
    updateAppearanceReference(img, trackInfo_.bbox);
    unverifiedFrames_ = 0;
    miss_ = 0;
    return true;
}

bool DeepTracker::isAppearanceReasonable(const cv::Mat &img,
                                         const DrOBB &bbox) const
{
    if (!trackerConfig_.targetManagement.enableAppearanceCheck ||
        appearanceHist_.empty())
    {
        return true;
    }

    cv::Mat hist;
    double grayStdDev = 0.0;
    double edgeDensity = 0.0;
    if (!computeAppearanceFeature(img, bbox, hist, grayStdDev, edgeDensity))
    {
        return false;
    }

    if (grayStdDev < trackerConfig_.targetManagement.minPatchStdDev ||
        edgeDensity < trackerConfig_.targetManagement.minEdgeDensity)
    {
        return false;
    }

    const double similarity =
        cv::compareHist(appearanceHist_, hist, cv::HISTCMP_CORREL);
    return similarity >=
           trackerConfig_.targetManagement.appearanceSimilarityThreshold;
}

void DeepTracker::resetTrackingState()
{
    is_tracked_ = false;
    age_ = 0;
    if (trackId_++ > 0xFFFFFFFF)
    {
        trackId_ = 0;
    }
    miss_ = 0;
    unverifiedFrames_ = 0;
    history_.clear();
    appearanceHist_.release();
    memset(&trackInfo_, 0, sizeof(trackInfo_));
    trackInfo_.reset = true;
}

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

        DrOBB selectedDetection{};
        if (!selectTargetDetection(detectObjList, img, nullptr,
                                   selectedDetection, matchedDetectId))
        {
            if (isSotDebugEnabled())
            {
                std::cerr << "[SOT] frame " << frameNum_
                          << " init skipped: no selectable detection, detections="
                          << detectObjList->numFilled << std::endl;
            }
            memset(&trackInfo_, 0, sizeof(trackInfo_));
            return trackInfo_;
        }

        if (!reinitializeFromDetection(img, selectedDetection))
        {
            if (isSotDebugEnabled())
            {
                std::cerr << "[SOT] frame " << frameNum_
                          << " init failed: tracker init returned error"
                          << std::endl;
            }
            memset(&trackInfo_, 0, sizeof(trackInfo_));
            return trackInfo_;
        }

        history_.clear();
        is_tracked_ = true;
        age_ = 1;
        is_good_track = true;
        trackInfo_.initialized = true;
        if (isSotDebugEnabled())
        {
            std::cerr << "[SOT] frame " << frameNum_
                      << " init from detection class="
                      << trackInfo_.bbox.class_id
                      << " score=" << trackInfo_.bbox.score
                      << " bbox=[" << trackInfo_.bbox.box.x0 << ","
                      << trackInfo_.bbox.box.y0 << ","
                      << trackInfo_.bbox.box.x1 << ","
                      << trackInfo_.bbox.box.y1 << "]" << std::endl;
        }
    }
    else
    {
        DrOBB prevBox = trackInfo_.bbox;
        NanotrackTRT *nanotrack =
            dynamic_cast<NanotrackTRT *>(trackerPtr_.get());
        const DrOBB &candidate =
            nanotrack != nullptr ? nanotrack->trackCandidate(img)
                                 : trackerPtr_->trackRgba(img);
        trackInfo_.bbox = candidate;

        const bool matchedDetection =
            findMatchedDetection(detectObjList, candidate, prevBox,
                                 matchedDetectId);
        const bool motionOk = isMotionReasonable(candidate, prevBox, img);
        const bool appearanceOk =
            matchedDetection || isAppearanceReasonable(img, candidate);
        const bool scoreOk =
            candidate.score >= trackerConfig_.targetManagement.highScoreThreshold ||
            (candidate.score >= trackerConfig_.targetManagement.lowScoreThreshold &&
             matchedDetection);
        const bool probationOk =
            age_ > trackerConfig_.targetManagement.probationAge ||
            matchedDetection;
        DrOBB selectedDetection{};
        uint32_t selectedDetectId = std::numeric_limits<uint32_t>::max();
        const bool hasTargetDetection =
            selectTargetDetection(detectObjList, img, &prevBox,
                                  selectedDetection, &selectedDetectId);
        if (isSotDebugEnabled())
        {
            std::cerr << "[SOT] frame " << frameNum_
                      << " candidate score=" << candidate.score
                      << " matched=" << matchedDetection
                      << " motion=" << motionOk
                      << " appearance=" << appearanceOk
                      << " scoreOk=" << scoreOk
                      << " probationOk=" << probationOk
                      << " hasDetection=" << hasTargetDetection
                      << " age=" << age_
                      << " miss=" << miss_ << std::endl;
        }

        if (scoreOk && motionOk && appearanceOk && probationOk)
        {
            const bool hasValidMatchedId =
                matchedDetection && matchedDetectId != nullptr &&
                detectObjList != nullptr &&
                *matchedDetectId < detectObjList->numFilled;
            const uint32_t nextUnverifiedFrames =
                hasValidMatchedId ? 0 : unverifiedFrames_ + 1;
            const uint32_t maxUnverified =
                trackerConfig_.targetManagement.maxUnverifiedFrames > 0
                    ? trackerConfig_.targetManagement.maxUnverifiedFrames
                    : trackerConfig_.targetManagement.requireDetectorEveryNFrames;
            const bool verifiedWindowOk =
                trackerConfig_.targetManagement.requireDetectorEveryNFrames == 0 ||
                maxUnverified == 0 || nextUnverifiedFrames <= maxUnverified;

            if (!verifiedWindowOk)
            {
                if (nanotrack != nullptr)
                {
                    nanotrack->rejectCandidate();
                }
                trackInfo_.bbox = prevBox;
                miss_++;
                is_good_track = false;
            }
            else
            {
                if (nanotrack != nullptr)
                {
                    nanotrack->commitCandidate();
                }

                if (hasValidMatchedId)
                {
                    const NvMOTObjToTrack &obj =
                        detectObjList->list[*matchedDetectId];
                    trackInfo_.bbox.class_id = obj.classId;
                    updateAppearanceReference(img, trackInfo_.bbox);
                }
                unverifiedFrames_ = nextUnverifiedFrames;
                miss_ = 0;
                age_++;
                is_good_track = true;
            }
        }
        else
        {
            if (nanotrack != nullptr)
            {
                nanotrack->rejectCandidate();
            }
            if (hasTargetDetection &&
                reinitializeFromDetection(img, selectedDetection))
            {
                if (matchedDetectId != nullptr)
                {
                    *matchedDetectId = selectedDetectId;
                }
                age_++;
                is_good_track = true;
                if (isSotDebugEnabled())
                {
                    std::cerr << "[SOT] frame " << frameNum_
                              << " candidate rejected, reinit from detection class="
                              << trackInfo_.bbox.class_id
                              << " score=" << trackInfo_.bbox.score
                              << std::endl;
                }
            }
            else
            {
                trackInfo_.bbox = prevBox;
                miss_++;
                is_good_track = false;
                if (isSotDebugEnabled())
                {
                    std::cerr << "[SOT] frame " << frameNum_
                              << " candidate rejected, no detection fallback"
                              << " miss=" << miss_ << std::endl;
                }
            }
        }

        if (miss_ > trackerConfig_.targetManagement.maxMiss)
        {
            resetTrackingState();
            return trackInfo_;
        }

        if (age_ < confirmAgeThreshold_)
        {
            if (isSotDebugEnabled())
            {
                std::cerr << "[SOT] frame " << frameNum_
                          << " hidden by confirmAge age=" << age_
                          << " confirmAge=" << confirmAgeThreshold_
                          << std::endl;
            }
            memset(&trackInfo_, 0, sizeof(trackInfo_));
            return trackInfo_;
        }

        if (!is_good_track)
        {
            if (isSotDebugEnabled())
            {
                std::cerr << "[SOT] frame " << frameNum_
                          << " hidden: not good track, miss=" << miss_
                          << std::endl;
            }
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

        // 仅在检测框已经连续无法验证当前跟踪时，才启用“中心过稳定”过滤。
        // 真实无人机悬停或云台随动时，目标中心可能长期稳定；如果每帧仍能被检测框匹配，
        // 不能按静止目标处理，否则会周期性 reset，打断下游按 object_id 累积的视频识别历史。
        if (enableTrackCenterStable_ && unverifiedFrames_ > 0)
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
                        if (isSotDebugEnabled())
                        {
                            std::cerr << "[SOT] frame " << frameNum_
                                      << " stable-center reset std=("
                                      << std_dev_cx << "," << std_dev_cy
                                      << ") threshold="
                                      << trackCenterStablePixelThreshold_
                                      << " unverified=" << unverifiedFrames_
                                      << " age=" << age_ << std::endl;
                        }
                        // 如果平均中心位置变化很小，并且稳定在非画面中心位置，认为是虚假跟踪
                        resetTrackingState();
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
