#include "Tracker.h"
#include <algorithm>
#include <cctype>
#include <iostream>

// 解析配置文件
int parseConfigFile(const char *pCustomConfigFilePath, TRACKER_CONFIG &trackerConfig)
{
    if (pCustomConfigFilePath == nullptr || strlen(pCustomConfigFilePath) == 0)
    {
        std::cerr << "Invalid custom config file path." << std::endl;
        return -1;
    }

    // 使用YAML库加载配置文件
    YAML::Node configyml;
    try
    {
        configyml = YAML::LoadFile(pCustomConfigFilePath);
    }
    catch (const YAML::Exception &e)
    {
        std::cerr << "Failed to parse config file: " << e.what() << std::endl;
        return -1;
    }
    if (!configyml)
    {
        std::cerr << "Failed to load config file: " << pCustomConfigFilePath << std::endl;
        return -1;
    }

    std::string key;
    // 解析配置文件内容
    if (!configyml["BaseConfig"])
    {
        std::cerr << "BaseConfig section not found in config file." << std::endl;
        return -1;
    }
    // 先初始化新增字段默认值
    trackerConfig.enableTrackCenterStable = true;
    trackerConfig.trackCenterStablePixelThreshold = 3;
    trackerConfig.nanotrack.mode = NANOTRACK_MODE_SPLIT;

    for (YAML::const_iterator itr = configyml["BaseConfig"].begin();
         itr != configyml["BaseConfig"].end(); ++itr)
    {
        key = itr->first.as<std::string>();
        if (key == "modelName")
        {
            trackerConfig.modelName = static_cast<MODEL_NAME>(itr->second.as<int>());
            if (trackerConfig.modelName < MODEL_SUTRACK || trackerConfig.modelName > MODEL_NANOTRACK)
            {
                std::cerr << "Invalid modelName in config file, set to default MixFormerV2" << std::endl;
                trackerConfig.modelName = MODEL_MIXFORMERV2; // 默认设置为 MixFormerV2
            }
        }
        else if (key == "enableTrackCenterStable")
        {
            trackerConfig.enableTrackCenterStable = itr->second.as<int>() != 0; // 0 视为 false 其它为 true
        }
        else if (key == "trackCenterStablePixelThreshold")
        {
            trackerConfig.trackCenterStablePixelThreshold = itr->second.as<uint32_t>();
            if (trackerConfig.trackCenterStablePixelThreshold == 0)
            {
                trackerConfig.trackCenterStablePixelThreshold = 3; // 防止为0
                std::cerr << "Invalid trackCenterStablePixelThreshold in config file, set to default 3" << std::endl;
            }
        }
        else
        {
            std::cerr << "Unknown key in config file: " << key << std::endl;
        }
    }

    if (!configyml["TargetManagement"])
    {
        std::cerr << "TargetManagement section not found in config file." << std::endl;
        return -1;
    }

    bool scoreThresholdSet = false;
    bool lowScoreThresholdSet = false;
    bool highScoreThresholdSet = false;
    for (YAML::const_iterator itr = configyml["TargetManagement"].begin();
         itr != configyml["TargetManagement"].end(); ++itr)
    {
        key = itr->first.as<std::string>();
        if (key == "expandFactor")
        {
            trackerConfig.targetManagement.expandFactor = itr->second.as<float>();
            if (trackerConfig.targetManagement.expandFactor <= 0.0f)
            {
                // 强制要求膨胀因子大于等于1
                trackerConfig.targetManagement.expandFactor = 1.0f;
                std::cerr << "Invalid expandFactor in config file, set to default 1.0" << std::endl;
            }
        }
        else if (key == "probationAge")
        {
            trackerConfig.targetManagement.probationAge = itr->second.as<uint16_t>();
            if (trackerConfig.targetManagement.probationAge < 1)
            {
                // 强制要求 probationAge 大于等于1
                trackerConfig.targetManagement.probationAge = 1;
                std::cerr << "Invalid probationAge in config file, set to default 1" << std::endl;
            }
        }
        else if (key == "maxMiss")
        {
            trackerConfig.targetManagement.maxMiss = itr->second.as<uint16_t>();
            // maxMiss is an unsigned value (uint16_t) and cannot be negative,
            // so no negative check is necessary; keep the parsed value as-is.
        }
        else if (key == "scoreThreshold")
        {
            scoreThresholdSet = true;
            trackerConfig.targetManagement.scoreThreshold = itr->second.as<float>();
            if (trackerConfig.targetManagement.scoreThreshold < 0.0f || trackerConfig.targetManagement.scoreThreshold > 1.0f)
            {
                trackerConfig.targetManagement.scoreThreshold = 0.3f; // 默认设置为0.3
                std::cerr << "Invalid scoreThreshold in config file, set to default 0.3" << std::endl;
            }
        }
        else if (key == "lowScoreThreshold")
        {
            lowScoreThresholdSet = true;
            trackerConfig.targetManagement.lowScoreThreshold = itr->second.as<float>();
            if (trackerConfig.targetManagement.lowScoreThreshold < 0.0f ||
                trackerConfig.targetManagement.lowScoreThreshold > 1.0f)
            {
                trackerConfig.targetManagement.lowScoreThreshold = 0.35f;
                std::cerr << "Invalid lowScoreThreshold in config file, set to default 0.35" << std::endl;
            }
        }
        else if (key == "highScoreThreshold")
        {
            highScoreThresholdSet = true;
            trackerConfig.targetManagement.highScoreThreshold = itr->second.as<float>();
            if (trackerConfig.targetManagement.highScoreThreshold < 0.0f ||
                trackerConfig.targetManagement.highScoreThreshold > 1.0f)
            {
                trackerConfig.targetManagement.highScoreThreshold = 0.65f;
                std::cerr << "Invalid highScoreThreshold in config file, set to default 0.65" << std::endl;
            }
        }
        else if (key == "iouThreshold")
        {
            trackerConfig.targetManagement.iouThreshold = itr->second.as<float>();
            if (trackerConfig.targetManagement.iouThreshold < 0.0f || trackerConfig.targetManagement.iouThreshold > 1.0f)
            {
                trackerConfig.targetManagement.iouThreshold = 0.5f; // 默认设置为0.5
                std::cerr << "Invalid iouThreshold in config file, set to default 0.5" << std::endl;
            }
        }
        else if (key == "trackBoxWidthThreshold")
        {
            trackerConfig.targetManagement.trackBoxWidthThreshold = itr->second.as<float>();
            if (trackerConfig.targetManagement.trackBoxWidthThreshold < 0.0f || trackerConfig.targetManagement.trackBoxWidthThreshold > 1.0f)
            {
                trackerConfig.targetManagement.trackBoxWidthThreshold = 0.3f; // 默认设置为0.3
                std::cerr << "Invalid trackBoxWidthThreshold in config file, set to default 0.3" << std::endl;
            }
        }
        else if (key == "trackBoxHeightThreshold")
        {
            trackerConfig.targetManagement.trackBoxHeightThreshold = itr->second.as<float>();
            if (trackerConfig.targetManagement.trackBoxHeightThreshold < 0.0f || trackerConfig.targetManagement.trackBoxHeightThreshold > 1.0f)
            {
                trackerConfig.targetManagement.trackBoxHeightThreshold = 0.3f; // 默认设置为0.3
                std::cerr << "Invalid trackBoxHeightThreshold in config file, set to default 0.3" << std::endl;
            }
        }
        else if (key == "maxTrackAge")
        {
            trackerConfig.targetManagement.maxTrackAge = itr->second.as<uint32_t>();
            if (trackerConfig.targetManagement.maxTrackAge < 1)
            {
                trackerConfig.targetManagement.maxTrackAge = 1; // 默认设置为1
                std::cerr << "Invalid maxTrackAge in config file, set to default 1" << std::endl;
            }
        }
        else if (key == "maxCenterJumpRatio")
        {
            trackerConfig.targetManagement.maxCenterJumpRatio = itr->second.as<float>();
            if (trackerConfig.targetManagement.maxCenterJumpRatio <= 0.0f)
            {
                trackerConfig.targetManagement.maxCenterJumpRatio = 3.0f;
                std::cerr << "Invalid maxCenterJumpRatio in config file, set to default 3.0" << std::endl;
            }
        }
        else if (key == "maxScaleChangeRatio")
        {
            trackerConfig.targetManagement.maxScaleChangeRatio = itr->second.as<float>();
            if (trackerConfig.targetManagement.maxScaleChangeRatio < 1.0f)
            {
                trackerConfig.targetManagement.maxScaleChangeRatio = 2.0f;
                std::cerr << "Invalid maxScaleChangeRatio in config file, set to default 2.0" << std::endl;
            }
        }
        else if (key == "maxAspectChangeRatio")
        {
            trackerConfig.targetManagement.maxAspectChangeRatio = itr->second.as<float>();
            if (trackerConfig.targetManagement.maxAspectChangeRatio < 1.0f)
            {
                trackerConfig.targetManagement.maxAspectChangeRatio = 2.0f;
                std::cerr << "Invalid maxAspectChangeRatio in config file, set to default 2.0" << std::endl;
            }
        }
        else if (key == "requireDetectorEveryNFrames")
        {
            trackerConfig.targetManagement.requireDetectorEveryNFrames =
                itr->second.as<uint32_t>();
        }
        else if (key == "maxUnverifiedFrames")
        {
            trackerConfig.targetManagement.maxUnverifiedFrames =
                itr->second.as<uint32_t>();
        }
        else if (key == "targetClassId")
        {
            trackerConfig.targetManagement.targetClassId = itr->second.as<int>();
        }
        else if (key == "enableAppearanceCheck")
        {
            trackerConfig.targetManagement.enableAppearanceCheck =
                itr->second.as<int>() != 0;
        }
        else if (key == "appearanceSimilarityThreshold")
        {
            trackerConfig.targetManagement.appearanceSimilarityThreshold =
                itr->second.as<float>();
            if (trackerConfig.targetManagement.appearanceSimilarityThreshold <
                    -1.0f ||
                trackerConfig.targetManagement.appearanceSimilarityThreshold >
                    1.0f)
            {
                trackerConfig.targetManagement.appearanceSimilarityThreshold =
                    0.25f;
                std::cerr << "Invalid appearanceSimilarityThreshold in config file, set to default 0.25" << std::endl;
            }
        }
        else if (key == "minPatchStdDev")
        {
            trackerConfig.targetManagement.minPatchStdDev = itr->second.as<float>();
            if (trackerConfig.targetManagement.minPatchStdDev < 0.0f)
            {
                trackerConfig.targetManagement.minPatchStdDev = 5.0f;
                std::cerr << "Invalid minPatchStdDev in config file, set to default 5.0" << std::endl;
            }
        }
        else if (key == "minEdgeDensity")
        {
            trackerConfig.targetManagement.minEdgeDensity = itr->second.as<float>();
            if (trackerConfig.targetManagement.minEdgeDensity < 0.0f ||
                trackerConfig.targetManagement.minEdgeDensity > 1.0f)
            {
                trackerConfig.targetManagement.minEdgeDensity = 0.001f;
                std::cerr << "Invalid minEdgeDensity in config file, set to default 0.001" << std::endl;
            }
        }
        else
        {
            std::cerr << "Unknown key in TargetManagement: " << key << std::endl;
        }
    }
    if (scoreThresholdSet && !lowScoreThresholdSet)
    {
        trackerConfig.targetManagement.lowScoreThreshold =
            trackerConfig.targetManagement.scoreThreshold;
    }
    if (scoreThresholdSet && !highScoreThresholdSet)
    {
        trackerConfig.targetManagement.highScoreThreshold =
            std::max(trackerConfig.targetManagement.scoreThreshold,
                     trackerConfig.targetManagement.highScoreThreshold);
    }
    if (trackerConfig.targetManagement.highScoreThreshold <
        trackerConfig.targetManagement.lowScoreThreshold)
    {
        trackerConfig.targetManagement.highScoreThreshold =
            trackerConfig.targetManagement.lowScoreThreshold;
        std::cerr << "highScoreThreshold is lower than lowScoreThreshold, align to lowScoreThreshold" << std::endl;
    }

    if (trackerConfig.modelName == MODEL_MIXFORMERV2)
    {
        YAML::Node mixformerNode;
        if (configyml["MixformerV2Config"])
        {
            mixformerNode = configyml["MixformerV2Config"];
        }
        else if (configyml["MixformerV2"])
        {
            mixformerNode = configyml["MixformerV2"];
        }

        if (mixformerNode)
        {
            if (mixformerNode["modelFilePath"])
            {
                trackerConfig.mixformerV2.modelFilePath = mixformerNode["modelFilePath"].as<std::string>();
            }

            if (mixformerNode["update_interval"])
            {
                int interval = mixformerNode["update_interval"].as<int>();
                if (interval <= 0)
                {
                    std::cerr << "Invalid update_interval in config file, set to default 200" << std::endl;
                }
                else
                {
                    trackerConfig.mixformerV2.updateInterval = interval;
                }
            }

            if (mixformerNode["max_score_decay"])
            {
                float decay = mixformerNode["max_score_decay"].as<float>();
                if (decay <= 0.0f || decay > 1.0f)
                {
                    std::cerr << "Invalid max_score_decay in config file, set to default 0.95" << std::endl;
                }
                else
                {
                    trackerConfig.mixformerV2.maxScoreDecay = decay;
                }
            }

            if (mixformerNode["template_update_score_threshold"])
            {
                float threshold =
                    mixformerNode["template_update_score_threshold"].as<float>();
                if (threshold < 0.0f || threshold > 1.0f)
                {
                    std::cerr << "Invalid template_update_score_threshold in config file, set to default 0.5" << std::endl;
                }
                else
                {
                    trackerConfig.mixformerV2.templateUpdateScoreThreshold =
                        threshold;
                }
            }
            if (mixformerNode["template_factor"])
            {
                float templateFactor =
                    mixformerNode["template_factor"].as<float>();
                if (templateFactor <= 0.0f)
                {
                    std::cerr << "Invalid template_factor in config file, set to default 2.0" << std::endl;
                }
                else
                {
                    trackerConfig.mixformerV2.templateFactor = templateFactor;
                }
            }

            if (mixformerNode["search_factor"])
            {
                float searchFactor =
                    mixformerNode["search_factor"].as<float>();
                if (searchFactor <= 0.0f)
                {
                    std::cerr << "Invalid search_factor in config file, set to default 4.0" << std::endl;
                }
                else
                {
                    trackerConfig.mixformerV2.searchFactor = searchFactor;
                }
            }
        }
    }

    YAML::Node nanotrackNode;
    if (configyml["NanotrackConfig"])
    {
        nanotrackNode = configyml["NanotrackConfig"];
    }
    else if (configyml["Nanotrack"])
    {
        nanotrackNode = configyml["Nanotrack"];
    }

    if (nanotrackNode)
    {
        std::string mode;
        if (nanotrackNode["nanotrack_mode"])
        {
            mode = nanotrackNode["nanotrack_mode"].as<std::string>();
        }
        else if (nanotrackNode["mode"])
        {
            mode = nanotrackNode["mode"].as<std::string>();
        }

        if (!mode.empty())
        {
            std::string mode_lower;
            mode_lower.resize(mode.size());
            std::transform(mode.begin(), mode.end(), mode_lower.begin(),
                           [](unsigned char c) { return std::tolower(c); });

            if (mode_lower == "merge")
            {
                trackerConfig.nanotrack.mode = NANOTRACK_MODE_MERGE;
            }
            else if (mode_lower == "split")
            {
                trackerConfig.nanotrack.mode = NANOTRACK_MODE_SPLIT;
            }
            else
            {
                std::cerr << "Invalid nanotrack_mode in config file, set to default split" << std::endl;
                trackerConfig.nanotrack.mode = NANOTRACK_MODE_SPLIT;
            }
        }

        if (nanotrackNode["nanotrack_merge_engine"])
        {
            trackerConfig.nanotrack.mergeEngine =
                nanotrackNode["nanotrack_merge_engine"].as<std::string>();
        }
        else if (nanotrackNode["merge_engine"])
        {
            trackerConfig.nanotrack.mergeEngine =
                nanotrackNode["merge_engine"].as<std::string>();
        }

        if (nanotrackNode["nanotrack_head_engine"])
        {
            trackerConfig.nanotrack.headEngine =
                nanotrackNode["nanotrack_head_engine"].as<std::string>();
        }
        else if (nanotrackNode["head_engine"])
        {
            trackerConfig.nanotrack.headEngine =
                nanotrackNode["head_engine"].as<std::string>();
        }

        if (nanotrackNode["nanotrack_backbone_engine"])
        {
            trackerConfig.nanotrack.backboneEngine =
                nanotrackNode["nanotrack_backbone_engine"].as<std::string>();
        }
        else if (nanotrackNode["backbone_engine"])
        {
            trackerConfig.nanotrack.backboneEngine =
                nanotrackNode["backbone_engine"].as<std::string>();
        }

        if (nanotrackNode["nanotrack_search_backbone_engine"])
        {
            trackerConfig.nanotrack.searchBackboneEngine =
                nanotrackNode["nanotrack_search_backbone_engine"].as<std::string>();
        }
        else if (nanotrackNode["search_backbone_engine"])
        {
            trackerConfig.nanotrack.searchBackboneEngine =
                nanotrackNode["search_backbone_engine"].as<std::string>();
        }

        if (nanotrackNode["nanotrack_exemplar_size"])
        {
            trackerConfig.nanotrack.exemplarSize =
                nanotrackNode["nanotrack_exemplar_size"].as<int>();
        }
        else if (nanotrackNode["exemplar_size"])
        {
            trackerConfig.nanotrack.exemplarSize =
                nanotrackNode["exemplar_size"].as<int>();
        }

        if (nanotrackNode["nanotrack_instance_size"])
        {
            trackerConfig.nanotrack.instanceSize =
                nanotrackNode["nanotrack_instance_size"].as<int>();
        }
        else if (nanotrackNode["instance_size"])
        {
            trackerConfig.nanotrack.instanceSize =
                nanotrackNode["instance_size"].as<int>();
        }

        if (nanotrackNode["window_influence"])
        {
            trackerConfig.nanotrack.windowInfluence =
                nanotrackNode["window_influence"].as<float>();
            if (trackerConfig.nanotrack.windowInfluence < 0.0f ||
                trackerConfig.nanotrack.windowInfluence > 1.0f)
            {
                trackerConfig.nanotrack.windowInfluence = -1.0f;
                std::cerr << "Invalid window_influence in config file, use NanoTrack default" << std::endl;
            }
        }
        if (nanotrackNode["penalty_k"])
        {
            trackerConfig.nanotrack.penaltyK =
                nanotrackNode["penalty_k"].as<float>();
            if (trackerConfig.nanotrack.penaltyK < 0.0f)
            {
                trackerConfig.nanotrack.penaltyK = -1.0f;
                std::cerr << "Invalid penalty_k in config file, use NanoTrack default" << std::endl;
            }
        }
        if (nanotrackNode["lr"])
        {
            trackerConfig.nanotrack.lr = nanotrackNode["lr"].as<float>();
            if (trackerConfig.nanotrack.lr < 0.0f ||
                trackerConfig.nanotrack.lr > 1.0f)
            {
                trackerConfig.nanotrack.lr = -1.0f;
                std::cerr << "Invalid lr in config file, use NanoTrack default" << std::endl;
            }
        }
        if (nanotrackNode["context_amount"])
        {
            trackerConfig.nanotrack.contextAmount =
                nanotrackNode["context_amount"].as<float>();
            if (trackerConfig.nanotrack.contextAmount <= 0.0f)
            {
                trackerConfig.nanotrack.contextAmount = -1.0f;
                std::cerr << "Invalid context_amount in config file, use NanoTrack default" << std::endl;
            }
        }

        if (trackerConfig.nanotrack.exemplarSize < 0)
        {
            std::cerr << "Invalid nanotrack_exemplar_size in config file, set to default" << std::endl;
            trackerConfig.nanotrack.exemplarSize = 0;
        }
        if (trackerConfig.nanotrack.instanceSize < 0)
        {
            std::cerr << "Invalid nanotrack_instance_size in config file, set to default" << std::endl;
            trackerConfig.nanotrack.instanceSize = 0;
        }
    }

    return 0;
}

NvMOTStatus NvMOT_Query(uint16_t customConfigFilePathSize,
                        char *pCustomConfigFilePath,
                        NvMOTQuery *pQuery)
{
    (void)customConfigFilePathSize;
    if (pQuery == nullptr)
        return NvMOTStatus_Error;
    TRACKER_CONFIG trackerConfig;
    // 解析自定义配置文件
    if (parseConfigFile(pCustomConfigFilePath, trackerConfig) != 0)
    {
        std::cerr << "Failed to parse custom config file: " << pCustomConfigFilePath << std::endl;
        return NvMOTStatus_Error;
    }

    /**  所有自定义跟踪器的必需配置。 */
    pQuery->computeConfig = NVMOTCOMP_CPU; // among {NVMOTCOMP_GPU, NVMOTCOMP_CPU}
    pQuery->numTransforms = 1;             // 0 for IOU and NvSORT tracker, 1 for NvDCF or NvDeepSORT tracker as they require the video frames
#ifdef __aarch64__
    pQuery->memType = NVBUF_MEM_SURFACE_ARRAY;
#else
    pQuery->memType = NVBUF_MEM_CUDA_UNIFIED;
#endif
    pQuery->batchMode = NvMOTBatchMode_Batch;          // batchMode must be set as NvMOTBatchMode_Batch
    pQuery->colorFormats[0] = NVBUF_COLOR_FORMAT_RGBA; // among {NVBUF_COLOR_FORMAT_NV12, NVBUF_COLOR_FORMAT_RGBA}
    pQuery->supportPastFrame = false;

    pQuery->maxTargetsPerStream = MAX_TARGETS_PER_STREAM; // Max number of targets stored for each stream

    /** 可选配置以设置其他功能。 */
    pQuery->maxShadowTrackingAge = 0;
    pQuery->outputReidTensor = false;  // 仅当低级跟踪器支持输出 reid 特性时设置为 true
    pQuery->reidFeatureSize = 0;
    pQuery->outputTrajectory = false;
    pQuery->outputVisibility = false;
    pQuery->outputFootLocation = false;
    pQuery->outputConvexHull = false;
    pQuery->maxConvexHullSize = 0;
    pQuery->outputTerminatedTracks = false;
    pQuery->maxTrajectoryBufferLength = 0;
    pQuery->outputShadowTracks = false;
    pQuery->contextHandle = nullptr;

    std::cout << "[Track Initialized]" << std::endl;
    return NvMOTStatus_OK;
}

NvMOTStatus NvMOT_Init(NvMOTConfig *pConfigIn,
                       NvMOTContextHandle *pContextHandle,
                       NvMOTConfigResponse *pConfigResponse)
{
    if (pConfigIn == nullptr || pContextHandle == nullptr ||
        pConfigResponse == nullptr)
        return NvMOTStatus_Error;
    *pContextHandle = nullptr;

    /// 用户定义的上下文类
    NvMOTContext *pContext = nullptr;

    /// 实例化用户定义的上下文
    try
    {
        pContext = new NvMOTContext(*pConfigIn, *pConfigResponse);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to initialize SOT context: " << e.what() << std::endl;
        pConfigResponse->summaryStatus = NvMOTConfigStatus_Error;
        return NvMOTStatus_Error;
    }

    if (pConfigResponse->summaryStatus != NvMOTConfigStatus_OK)
    {
        delete pContext;
        return NvMOTStatus_Error;
    }

    /// 将指针作为上下文句柄传递
    *pContextHandle = pContext;

    return NvMOTStatus_OK;
}

void NvMOT_DeInit(NvMOTContextHandle contextHandle)
{
    if (contextHandle == nullptr)
        return;
    delete contextHandle;
}

NvMOTStatus NvMOT_Process(NvMOTContextHandle contextHandle,
                          NvMOTProcessParams *pParams,
                          NvMOTTrackedObjBatch *pTrackedObjectsBatch)
{
    if (contextHandle == nullptr || pParams == nullptr ||
        pTrackedObjectsBatch == nullptr)
        return NvMOTStatus_Error;
    /// 使用上下文中的用户定义方法处理给定的视频帧，并生成输出
    return contextHandle->processFrame(pParams, pTrackedObjectsBatch);
}

NvMOTStatus NvMOT_RetrieveMiscData(NvMOTContextHandle contextHandle,
                                   NvMOTProcessParams *pParams,
                                   NvMOTTrackerMiscData *pTrackerMiscData)
{
    if (contextHandle == nullptr || pParams == nullptr ||
        pTrackerMiscData == nullptr)
        return NvMOTStatus_Error;
    /// 如果有，请检索过去帧的数据
    return contextHandle->retrieveMiscData(pParams, pTrackerMiscData);
}

NvMOTStatus NvMOT_RemoveStreams(NvMOTContextHandle contextHandle,
                                NvMOTStreamId streamIdMask)
{
    if (contextHandle == nullptr)
        return NvMOTStatus_Error;
    /// 从低级跟踪器上下文中删除指定的视频流
    return contextHandle->removeStream(streamIdMask);
}
