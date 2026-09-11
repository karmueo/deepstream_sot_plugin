#include "Tracker.h"
#include <algorithm>
#include <cctype>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <utility>

static bool isSotDebugEnabled()
{
    const char *env = std::getenv("SOT_DEBUG");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool isValidTrackedObjList(const NvMOTTrackedObjList *trackedObjList)
{
    // 检查指针是否为空
    if (trackedObjList == nullptr)
    {
        std::cerr << "Error: trackedObjList is nullptr!" << std::endl;
        return false;
    }

    if (trackedObjList->list == nullptr)
    {
        std::cerr << "Error: trackedObjList->list is nullptr!" << std::endl;
        return false;
    }

    // 所有检查通过，对象有效
    return true;
}

static NvMOTObjToTrack *extractMatchedDetection(NvMOTFrame                *frame,
                                                uint32_t                   matchedDetectId)
{
    if (frame == nullptr)
    {
        return nullptr;
    }

    NvMOTObjToTrackList &objectsIn = frame->objectsIn;

    if (objectsIn.list == nullptr || objectsIn.numFilled == 0)
    {
        return nullptr;
    }

    if (matchedDetectId != std::numeric_limits<uint32_t>::max() &&
        matchedDetectId < objectsIn.numFilled)
    {
        return &objectsIn.list[matchedDetectId];
    }
    return nullptr;
}

NvMOTContext::NvMOTContext(const NvMOTConfig   &configIn,
                           NvMOTConfigResponse &configResponse)
{
    configResponse.summaryStatus = NvMOTConfigStatus_OK;
    // 解析配置文件
    // 先为新增可配置参数设定默认值，防止解析失败时未初始化
    trackerConfig_.enableTrackCenterStable = true;
    trackerConfig_.trackCenterStablePixelThreshold = 3;
    if (parseConfigFile(configIn.customConfigFilePath, trackerConfig_) != 0)
    {
        std::cerr << "Failed to parse custom config file: "
                  << configIn.customConfigFilePath << std::endl;
        configResponse.summaryStatus = NvMOTConfigStatus_Error;
        configResponse.customConfigStatus = NvMOTConfigStatus_Error;
        return;
    }
    configResponse.computeStatus = NvMOTConfigStatus_OK;
    configResponse.transformBatchStatus = NvMOTConfigStatus_OK;
    configResponse.miscConfigStatus = NvMOTConfigStatus_OK;
    configResponse.customConfigStatus = NvMOTConfigStatus_OK;

    if (!trackerConfig_.mixformerV2.modelFilePath.empty())
    {
        std::ifstream ifs(trackerConfig_.mixformerV2.modelFilePath);
        if (!ifs.good())
        {
            std::cerr << "Warning: model file not found at "
                      << trackerConfig_.mixformerV2.modelFilePath << "." << std::endl;
        }
    }
    const char *debug = std::getenv("SOT_DEBUG");
    debugLogging_ = debug != nullptr && debug[0] != '\0' && debug[0] != '0';
    const char *gstDebug = std::getenv("GST_DEBUG");
    if (!debugLogging_ && gstDebug != nullptr)
    {
        const char *sotLevel = std::strstr(gstDebug, "sot:");
        debugLogging_ = (sotLevel != nullptr && std::atoi(sotLevel + 4) >= 4) ||
                        (std::isdigit(static_cast<unsigned char>(gstDebug[0])) &&
                         std::atoi(gstDebug) >= 4);
    }
}

NvMOTContext::~NvMOTContext() {}

NvMOTStatus
NvMOTContext::processFrame(const NvMOTProcessParams *params,
                           NvMOTTrackedObjBatch     *pTrackedObjectsBatch)
{
    if (!params || !pTrackedObjectsBatch || params->numFrames == 0)
    {
        return params && pTrackedObjectsBatch ? NvMOTStatus_OK : NvMOTStatus_Invalid_Path;
    }
    if (!params->frameList || !pTrackedObjectsBatch->list)
        return NvMOTStatus_Error;

    const uint32_t resetCount = std::min(params->numFrames,
                                         pTrackedObjectsBatch->numAllocated);
    for (uint32_t i = 0; i < resetCount; ++i)
    {
        NvMOTTrackedObjList &output = pTrackedObjectsBatch->list[i];
        output.streamID = params->frameList[i].streamID;
        output.frameNum = params->frameList[i].frameNum;
        output.valid = false;
        output.numFilled = 0;
    }
    if (pTrackedObjectsBatch->numAllocated < params->numFrames)
        return NvMOTStatus_Error;

    NvMOTStatus batchStatus = NvMOTStatus_OK;
    for (uint streamIdx = 0; streamIdx < params->numFrames; streamIdx++)
    {
        NvMOTFrame *frame = &params->frameList[streamIdx];
        NvMOTTrackedObjList *trackedObjList =
            &pTrackedObjectsBatch->list[streamIdx];
        trackedObjList->streamID = frame->streamID;
        trackedObjList->frameNum = frame->frameNum;
        trackedObjList->valid = false;
        trackedObjList->numFilled = 0;
        if (isValidTrackedObjList(trackedObjList) == false ||
            trackedObjList->numAllocated < 1)
        {
            batchStatus = NvMOTStatus_Error;
            continue;
        }

        if (!frame->bufferList || frame->numBuffers < 1 ||
            frame->bufferList[0] == nullptr)
        {
            std::cerr << "SOT: missing transform buffer for stream "
                      << frame->streamID << std::endl;
            batchStatus = NvMOTStatus_Error;
            continue;
        }

        NvBufSurfaceParams *bufferParams = frame->bufferList[0];
        // NVBUF_MEM_SURFACE_ARRAY 时 dataPtr 不可直接使用，优先用 mappedAddr
        void *rgbaPtr = bufferParams->mappedAddr.addr[0] != nullptr
                            ? bufferParams->mappedAddr.addr[0]
                            : bufferParams->dataPtr;
        if (rgbaPtr == nullptr)
        {
            std::cerr << "Invalid RGBA buffer pointer (dataPtr/mappedAddr is null)" << std::endl;
            batchStatus = NvMOTStatus_Error;
            continue;
        }
        cv::Mat rgbaFrame(bufferParams->height, bufferParams->width, CV_8UC4,
                          rgbaPtr, bufferParams->pitch);

        auto trackerIt = trackers_.find(frame->streamID);
        if (trackerIt == trackers_.end())
        {
            try
            {
                auto tracker = std::make_unique<DeepTracker>(
                    trackerConfig_.mixformerV2.modelFilePath, trackerConfig_);
                trackerIt = trackers_.emplace(frame->streamID,
                                               std::move(tracker)).first;
            }
            catch (const std::exception &e)
            {
                std::cerr << "SOT: unable to create tracker for stream "
                          << frame->streamID << ": " << e.what() << std::endl;
                batchStatus = NvMOTStatus_Error;
                continue;
            }
        }
        DeepTracker *tracker = trackerIt->second.get();

        TrackInfo trackInfo;
        uint32_t  matchedDetectId = std::numeric_limits<uint32_t>::max();
        try
        {
            trackInfo = tracker->update(rgbaFrame, &frame->objectsIn,
                                        frame->frameNum, &matchedDetectId);
        }
        catch (const std::exception &e)
        {
            std::cerr << "SOT: tracker failed for stream " << frame->streamID
                      << ": " << e.what() << std::endl;
            trackers_.erase(frame->streamID);
            fpsState_.erase(frame->streamID);
            batchStatus = NvMOTStatus_Error;
            continue;
        }

        // 计算并显示FPS（每3秒更新一次）
        auto now = std::chrono::steady_clock::now();
        FpsState &fpsState = fpsState_[frame->streamID];
        if (fpsState.lastUpdate.time_since_epoch().count() == 0)
        {
            fpsState.lastUpdate = now;
        }
        fpsState.frameCount += 1;
        std::chrono::duration<double> elapsed = now - fpsState.lastUpdate;
        if (elapsed.count() >= fpsUpdateInterval_.count())
        {
            fpsState.fps = fpsState.frameCount / elapsed.count();
            fpsState.frameCount = 0;
            fpsState.lastUpdate = now;
            fpsState.hasValue = true;
        }

        NvMOTObjToTrack *associatedObjectIn =
            extractMatchedDetection(frame, matchedDetectId);

        NvMOTTrackedObj *trackedObjs = trackedObjList->list;
        // 单目标跟踪，所以只要第一个目标
        NvMOTTrackedObj *trackedObj = &trackedObjs[0];

        // 如果跟踪上了，给NvMOTTrackedObj赋值
        const bool outputTrack =
            tracker->isTracked() &&
            trackInfo.age >= trackerConfig_.targetManagement.probationAge &&
            trackInfo.bbox.box.w > 0.f && trackInfo.bbox.box.h > 0.f;
        if (isSotDebugEnabled())
        {
            std::cerr << "[SOT] frame " << frame->frameNum
                      << " NvMOT output=" << outputTrack
                      << " isTracked=" << tracker->isTracked()
                      << " age=" << trackInfo.age
                      << " probation="
                      << trackerConfig_.targetManagement.probationAge
                      << " detections=" << frame->objectsIn.numFilled
                      << " matchedDetectId=" << matchedDetectId
                      << " score=" << trackInfo.bbox.score
                      << std::endl;
        }
        if (outputTrack)
        {
            // 按 expandFactor 膨胀
            float new_w = trackInfo.bbox.box.w *
                          trackerConfig_.targetManagement.expandFactor;
            float new_h = trackInfo.bbox.box.h *
                          trackerConfig_.targetManagement.expandFactor;

            float new_x0 = trackInfo.bbox.box.cx - new_w / 2.0f;
            float new_y0 = trackInfo.bbox.box.cy - new_h / 2.0f;
            if (new_x0 < 0)
            {
                new_x0 = 0;
            }
            if (new_y0 < 0)
            {
                new_y0 = 0;
            }
            if (new_x0 + new_w > bufferParams->width)
            {
                new_w = bufferParams->width - new_x0;
            }
            if (new_y0 + new_h > bufferParams->height)
            {
                new_h = bufferParams->height - new_y0;
            }

            NvMOTRect motRect{new_x0, new_y0, new_w, new_h};

            // 更新跟踪对象信息
            trackedObj->classId = trackInfo.bbox.class_id;
            trackedObj->trackingId = trackInfo.trackId;
            trackedObj->bbox = motRect;
            trackedObj->confidence = trackInfo.bbox.score;
            trackedObj->age = trackInfo.age;
            trackedObj->associatedObjectIn = associatedObjectIn;
            // trackedObj->associatedObjectIn = objectToTrack_;
            // trackedObj->associatedObjectIn->doTracking = true;

            // trackedObjList->streamID = frame->streamID;
            trackedObjList->frameNum = frame->frameNum;
            trackedObjList->valid = true;
            trackedObjList->numFilled = 1;
        }
        else
        {
            // 取消跟踪，清空对象列表
            // trackedObjList->streamID = frame->streamID;
            trackedObjList->frameNum = frame->frameNum;
            trackedObjList->numFilled = 0;
            trackedObjList->valid = false;
            if (trackedObj != nullptr)
            {
                trackedObj->associatedObjectIn = nullptr;
            }
        }

        if (debugLogging_ && (frame->frameNum % 30 == 0 ||
                              trackInfo.initialized || trackInfo.reset))
        {
            std::cout << "[SOT] stream=" << frame->streamID
                      << " frame=" << frame->frameNum
                      << " id=" << trackInfo.trackId
                      << " age=" << trackInfo.age
                      << " miss=" << trackInfo.miss
                      << " bbox=" << trackInfo.bbox.box.x0 << ','
                      << trackInfo.bbox.box.y0 << ',' << trackInfo.bbox.box.w
                      << ',' << trackInfo.bbox.box.h
                      << " score=" << trackInfo.bbox.score
                      << " init=" << trackInfo.initialized
                      << " reset=" << trackInfo.reset
                      << " tracked=" << tracker->isTracked() << std::endl;
        }
    }

    return batchStatus;
}

NvMOTStatus
NvMOTContext::retrieveMiscData(const NvMOTProcessParams *params,
                               NvMOTTrackerMiscData     *pTrackerMiscData)
{
    (void)params;
    (void)pTrackerMiscData;
    return NvMOTStatus_OK;
}

NvMOTStatus NvMOTContext::removeStream(const NvMOTStreamId streamIdMask)
{
    for (auto it = trackers_.begin(); it != trackers_.end();)
    {
        if ((it->first & streamIdMask) == streamIdMask)
            it = trackers_.erase(it);
        else
            ++it;
    }
    for (auto it = fpsState_.begin(); it != fpsState_.end();)
    {
        if ((it->first & streamIdMask) == streamIdMask)
            it = fpsState_.erase(it);
        else
            ++it;
    }
    return NvMOTStatus_OK;
}
