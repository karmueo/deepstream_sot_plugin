//
// Created by Mayur Kulkarni on 11/11/21.
//

#ifndef DNSTARPROD_TRACKER_H
#define DNSTARPROD_TRACKER_H

#include "nvdstracker.h"
#include <opencv2/opencv.hpp>
// #include "mixformer_trt.h"
#include <memory>
#include "deepTracker.h"
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <unordered_map>

#define MAX_TARGETS_PER_STREAM 512

using namespace std;

int parseConfigFile(const char *pCustomConfigFilePath, TRACKER_CONFIG &trackerConfig);

/**
 * @brief Context for input video streams
 *
 * The stream context holds all necessary state to perform multi-object tracking
 * within the stream.
 *
 */
class NvMOTContext
{
public:
    NvMOTContext(const NvMOTConfig &configIn, NvMOTConfigResponse &configResponse);

    ~NvMOTContext();

    /**
     * @brief Process a batch of frames
     *
     * Internal implementation of NvMOT_Process()
     *
     * @param [in] pParam Pointer to parameters for the frame to be processed
     * @param [out] pTrackedObjectsBatch Pointer to object tracks output
     */
    NvMOTStatus processFrame(const NvMOTProcessParams *params,
                             NvMOTTrackedObjBatch *pTrackedObjectsBatch);
    /**
     * @brief Output the miscellaneous data if there are
     *
     *  Internal implementation of retrieveMiscData()
     *
     * @param [in] pParam Pointer to parameters for the frame to be processed
     * @param [out] pTrackerMiscData Pointer to miscellaneous data output
     */
    NvMOTStatus retrieveMiscData(const NvMOTProcessParams *params,
                                 NvMOTTrackerMiscData *pTrackerMiscData);
    /**
     * @brief Terminate trackers and release resources for a stream when the stream is removed
     *
     *  Internal implementation of NvMOT_RemoveStreams()
     *
     * @param [in] streamIdMask removed stream ID
     */
    NvMOTStatus removeStream(const NvMOTStreamId streamIdMask);

protected:
    std::shared_ptr<DeepTracker> tracker_;
    int tmpId_ = 0;
    TRACKER_CONFIG trackerConfig_; // 跟踪器配置
    struct FpsState
    {
        std::chrono::steady_clock::time_point lastUpdate{};
        uint32_t frameCount = 0;
        double fps = 0.0;
        bool hasValue = false;
    };
    std::unordered_map<NvMOTStreamId, FpsState> fpsState_;
    std::chrono::seconds fpsUpdateInterval_{3};
};

#endif // DNSTARPROD_TRACKER_H
