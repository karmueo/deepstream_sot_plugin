#include "Tracker.h"
#include <cuda_runtime.h>
#include <fstream>
#include <limits>

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

    if (!trackerConfig_.modelFilePath.empty())
    {
        std::ifstream ifs(trackerConfig_.modelFilePath);
        if (!ifs.good())
        {
            std::cerr << "Warning: model file not found at "
                      << trackerConfig_.modelFilePath << "." << std::endl;
        }
    }
    tracker_ = std::make_shared<DeepTracker>(trackerConfig_.modelFilePath,
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
            delete trackedObjList->list;
            trackedObjList->list = new NvMOTTrackedObj[1];
        }

        NvBufSurfaceParams *bufferParams = frame->bufferList[0];
        cv::Mat bgraFrame(bufferParams->height, bufferParams->width, CV_8UC4,
                          bufferParams->dataPtr);

        TrackInfo trackInfo;
        uint32_t  matchedDetectId = std::numeric_limits<uint32_t>::max();
        trackInfo = tracker_->update(bgraFrame, &frame->objectsIn,
                                     frame->frameNum, &matchedDetectId);

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
            if (matchedDetectId != std::numeric_limits<uint32_t>::max() &&
                frame->objectsIn.list != nullptr &&
                matchedDetectId < frame->objectsIn.numFilled)
            {
                trackedObj->associatedObjectIn =
                    &frame->objectsIn.list[matchedDetectId];
            }
            else
            {
                trackedObj->associatedObjectIn = nullptr;
            }
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
