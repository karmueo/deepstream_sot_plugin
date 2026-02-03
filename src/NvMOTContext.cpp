#include "Tracker.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <fstream>
#include <limits>
#include <utility>

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
        if (matchedDetectId != 0)
        {
            std::swap(objectsIn.list[0], objectsIn.list[matchedDetectId]);
        }

        objectsIn.numFilled = 1;
        return &objectsIn.list[0];
    }

    objectsIn.numFilled = 0;
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
    }

    if (!trackerConfig_.mixformerV2.modelFilePath.empty())
    {
        std::ifstream ifs(trackerConfig_.mixformerV2.modelFilePath);
        if (!ifs.good())
        {
            std::cerr << "Warning: model file not found at "
                      << trackerConfig_.mixformerV2.modelFilePath << "." << std::endl;
        }
    }
    tracker_ = std::make_shared<DeepTracker>(trackerConfig_.mixformerV2.modelFilePath,
                                             trackerConfig_);
}

NvMOTContext::~NvMOTContext() {}

NvMOTStatus
NvMOTContext::processFrame(const NvMOTProcessParams *params,
                           NvMOTTrackedObjBatch     *pTrackedObjectsBatch)
{
    cv::Mat in_mat;

    if (!params || params->numFrames <= 0)
    {
        return NvMOTStatus_OK;
    }

    for (uint streamIdx = 0; streamIdx < pTrackedObjectsBatch->numFilled;
         streamIdx++)
    {
        NvMOTTrackedObjList *trackedObjList =
            &pTrackedObjectsBatch->list[streamIdx];
        if (isValidTrackedObjList(trackedObjList) == false)
        {
            continue;
        }

        NvMOTFrame *frame = &params->frameList[streamIdx];

        if (frame->bufferList[0] == nullptr)
        {
            std::cout << "frame->bufferList[0] is nullptr" << std::endl;
            continue;
        }

        if (trackedObjList->numAllocated != 1)
        {
            // Reallocate memory space
            delete[] trackedObjList->list;
            trackedObjList->list = new NvMOTTrackedObj[1];
        }

        NvBufSurfaceParams *bufferParams = frame->bufferList[0];
        // NVBUF_MEM_SURFACE_ARRAY 时 dataPtr 不可直接使用，优先用 mappedAddr
        void *rgbaPtr = bufferParams->mappedAddr.addr[0] != nullptr
                            ? bufferParams->mappedAddr.addr[0]
                            : bufferParams->dataPtr;
        if (rgbaPtr == nullptr)
        {
            std::cerr << "Invalid RGBA buffer pointer (dataPtr/mappedAddr is null)" << std::endl;
            continue;
        }
        cv::Mat rgbaFrame(bufferParams->height, bufferParams->width, CV_8UC4,
                          rgbaPtr, bufferParams->pitch);
        cv::Mat bgrFrame;
        // 跟踪器输入为 3 通道 BGR，DeepStream 输出为 RGBA，需要转换
        cv::cvtColor(rgbaFrame, bgrFrame, cv::COLOR_RGBA2BGR);

        TrackInfo trackInfo;
        uint32_t  matchedDetectId = std::numeric_limits<uint32_t>::max();
        trackInfo = tracker_->update(bgrFrame, &frame->objectsIn,
                                     frame->frameNum, &matchedDetectId);

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
        if (tracker_->isTracked() &&
            trackInfo.age > trackerConfig_.targetManagement.probationAge)
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
            trackedObjList->list = trackedObjs;
            trackedObjList->numFilled = 1;
            trackedObjList->numAllocated = 1;
        }
        else
        {
            // 取消跟踪，清空对象列表
            // trackedObjList->streamID = frame->streamID;
            trackedObjList->frameNum = frame->frameNum;
            trackedObjList->numFilled = 0;
            trackedObjList->valid = false;
            trackedObjList->numAllocated = 1;
            if (trackedObj != nullptr)
            {
                trackedObj->associatedObjectIn = nullptr;
            }
            if (frame->objectsIn.list != nullptr)
            {
                frame->objectsIn.numFilled = 0;
            }
        }
    }

    return NvMOTStatus_OK;
}

NvMOTStatus
NvMOTContext::retrieveMiscData(const NvMOTProcessParams *params,
                               NvMOTTrackerMiscData     *pTrackerMiscData)
{
    (void)params;
    (void)pTrackerMiscData;
    /* std::set<NvMOTStreamId> videoStreamIdList;
    for (NvMOTStreamId streamInd = 0; streamInd < params->numFrames;
    streamInd++)
    {
        videoStreamIdList.insert(params->frameList[streamInd].streamID);
    }

    for (NvMOTStreamId streamInd = 0; streamInd < params->numFrames;
    streamInd++)
    {
        if (pTrackerMiscData && pTrackerMiscData->pPastFrameObjBatch)
        {
            tracker_->updatePastFrameObjBatch(pTrackerMiscData->pPastFrameObjBatch);
        }
    } */

    return NvMOTStatus_OK;
}

NvMOTStatus NvMOTContext::removeStream(const NvMOTStreamId streamIdMask)
{
    (void)streamIdMask;
    return NvMOTStatus_OK;
}
