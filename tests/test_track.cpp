#if __has_include(<catch2/catch_all.hpp>)
#include <catch2/catch_all.hpp>
#else
#include <catch2/catch.hpp>
#endif

#include "nanotrack_trt.h"
#include "mixformerv2_trt.h"
#include "Tracker.h"

#include <opencv2/opencv.hpp>

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>

#ifndef TRACK_TEST_CONFIG_PATH
#define TRACK_TEST_CONFIG_PATH "tests/track_test_config.yml"
#endif
#ifndef TRACK_TEST_REPO_ROOT
#define TRACK_TEST_REPO_ROOT "."
#endif

namespace {
bool fileExists(const std::string &path)
{
    std::ifstream file(path);
    return file.good();
}

std::string dirName(const std::string &path)
{
    const std::string::size_type pos = path.find_last_of('/');
    if (pos == std::string::npos)
    {
        return ".";
    }
    return path.substr(0, pos);
}

std::string resolvePath(const std::string &base_dir, const std::string &path)
{
    if (path.empty() || path[0] == '/')
    {
        return path;
    }
    const std::string candidate = base_dir + "/" + path;
    if (fileExists(candidate))
    {
        return candidate;
    }
    const std::string fallback =
        std::string(TRACK_TEST_REPO_ROOT) + "/" + path;
    if (fileExists(fallback))
    {
        return fallback;
    }
    return candidate;
}
} // namespace

TEST_CASE("NanotrackTRT keeps stable bbox on identical frames")
{
    const char *env_path = std::getenv("TRACK_TEST_CONFIG");
    std::string config_path =
        env_path != nullptr ? std::string(env_path)
                            : std::string(TRACK_TEST_CONFIG_PATH);

    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    REQUIRE(fs.isOpened());

    std::string image_path;
    std::string nanotrack_mode = "merge";
    std::string merge_engine;
    std::string head_engine;
    std::string backbone_engine;
    std::string search_backbone_engine;
    if (!fs["nanotrack_mode"].empty())
    {
        fs["nanotrack_mode"] >> nanotrack_mode;
    }
    fs["nanotrack_merge_engine"] >> merge_engine;
    fs["nanotrack_head_engine"] >> head_engine;
    fs["nanotrack_backbone_engine"] >> backbone_engine;
    fs["nanotrack_search_backbone_engine"] >> search_backbone_engine;
    fs["image_path"] >> image_path;

    REQUIRE(!image_path.empty());
    if (nanotrack_mode != "merge" && nanotrack_mode != "split")
    {
        FAIL("nanotrack_mode 仅支持 merge 或 split");
    }

    if (nanotrack_mode == "merge")
    {
        REQUIRE(!merge_engine.empty());
    }
    else
    {
        REQUIRE(!head_engine.empty());
        REQUIRE(!backbone_engine.empty());
    }

    cv::FileNode bbox_node = fs["init_bbox"];
    REQUIRE(bbox_node.isSeq());
    REQUIRE(bbox_node.size() == 4);

    float x0 = 0.f;
    float y0 = 0.f;
    float x1 = 0.f;
    float y1 = 0.f;
    bbox_node[0] >> x0;
    bbox_node[1] >> y0;
    bbox_node[2] >> x1;
    bbox_node[3] >> y1;

    int loop_count = 10;
    float max_abs_error = 1.0f;
    int exemplar_size = 0;
    int instance_size = 0;

    if (!fs["loop_count"].empty())
    {
        fs["loop_count"] >> loop_count;
    }
    if (!fs["max_abs_error"].empty())
    {
        fs["max_abs_error"] >> max_abs_error;
    }
    if (!fs["nanotrack_exemplar_size"].empty())
    {
        fs["nanotrack_exemplar_size"] >> exemplar_size;
    }
    if (!fs["nanotrack_instance_size"].empty())
    {
        fs["nanotrack_instance_size"] >> instance_size;
    }

    const std::string base_dir = dirName(config_path);
    image_path = resolvePath(base_dir, image_path);
    merge_engine = resolvePath(base_dir, merge_engine);
    head_engine = resolvePath(base_dir, head_engine);
    backbone_engine = resolvePath(base_dir, backbone_engine);
    if (!search_backbone_engine.empty())
    {
        search_backbone_engine = resolvePath(base_dir, search_backbone_engine);
    }

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    REQUIRE(!img.empty());

    std::unique_ptr<NanotrackTRT> tracker;
    if (nanotrack_mode == "merge")
    {
        tracker = std::make_unique<NanotrackTRT>(merge_engine);
    }
    else
    {
        tracker = std::make_unique<NanotrackTRT>(head_engine, backbone_engine,
                                                 search_backbone_engine);
    }
    if (exemplar_size > 0)
    {
        tracker->setExemplarSize(exemplar_size);
    }
    if (instance_size > 0)
    {
        tracker->setInstanceSize(instance_size);
    }

    DrOBB bbox;
    bbox.box.x0 = x0;
    bbox.box.y0 = y0;
    bbox.box.x1 = x1;
    bbox.box.y1 = y1;
    bbox.class_id = 0;

    REQUIRE(tracker->init(img, bbox) == 0);

    for (int i = 0; i < loop_count; ++i)
    {
        const DrOBB &tracked = tracker->track(img);
        REQUIRE(std::abs(tracked.box.x0 - x0) <= max_abs_error);
        REQUIRE(std::abs(tracked.box.y0 - y0) <= max_abs_error);
        REQUIRE(std::abs(tracked.box.x1 - x1) <= max_abs_error);
        REQUIRE(std::abs(tracked.box.y1 - y1) <= max_abs_error);
    }
}

TEST_CASE("NvMOTContext preserves client-owned batch and input lists")
{
    std::string config_path =
        std::string(TRACK_TEST_REPO_ROOT) + "/config_sot.yml";
    NvMOTConfig config{};
    config.maxStreams = 2;
    config.customConfigFilePath = &config_path[0];
    config.customConfigFilePathSize =
        static_cast<uint16_t>(config_path.size() + 1);
    NvMOTConfigResponse response{};
    NvMOTContext context(config, response);
    REQUIRE(response.summaryStatus == NvMOTConfigStatus_OK);

    cv::Mat bgr = cv::imread(
        std::string(TRACK_TEST_REPO_ROOT) + "/tests/test_image.jpg",
        cv::IMREAD_COLOR);
    REQUIRE(!bgr.empty());
    cv::Mat rgba;
    cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);

    NvBufSurfaceParams surface{};
    surface.width = static_cast<uint32_t>(rgba.cols);
    surface.height = static_cast<uint32_t>(rgba.rows);
    surface.pitch = static_cast<uint32_t>(rgba.step);
    surface.dataPtr = rgba.data;
    NvBufSurfaceParams *buffer_list[] = {&surface};

    NvMOTObjToTrack detections[2]{};
    detections[0].classId = 1;
    detections[0].bbox = {850.f, 500.f, 100.f, 80.f};
    detections[0].confidence = 0.9f;
    detections[1] = detections[0];
    detections[1].bbox.x = 400.f;

    NvMOTFrame frames[2]{};
    for (uint32_t i = 0; i < 2; ++i)
    {
        frames[i].streamID = static_cast<NvMOTStreamId>(i) << 32;
        frames[i].frameNum = 1;
        frames[i].numBuffers = 1;
        frames[i].bufferList = buffer_list;
        frames[i].objectsIn.list = &detections[i];
        frames[i].objectsIn.numAllocated = 1;
        frames[i].objectsIn.numFilled = 1;
    }
    const NvMOTObjToTrack input_copy[2] = {detections[0], detections[1]};

    NvMOTTrackedObj output_slots[2]{};
    NvMOTTrackedObjList output_lists[2]{};
    for (uint32_t i = 0; i < 2; ++i)
    {
        output_lists[i].list = &output_slots[i];
        output_lists[i].numAllocated = 1;
        output_lists[i].numFilled = 1;
        output_lists[i].valid = true;
    }
    NvMOTTrackedObjBatch output_batch{};
    output_batch.list = output_lists;
    output_batch.numAllocated = 2;
    output_batch.numFilled = 2;
    NvMOTProcessParams params{2, frames};

    NvMOTTrackedObjList *const original_batch_list = output_batch.list;
    NvMOTTrackedObj *const original_stream0_list = output_lists[0].list;
    NvMOTTrackedObj *const original_stream1_list = output_lists[1].list;
    REQUIRE(context.processFrame(&params, &output_batch) == NvMOTStatus_OK);
    REQUIRE(output_batch.list == original_batch_list);
    REQUIRE(output_lists[0].list == original_stream0_list);
    REQUIRE(output_lists[1].list == original_stream1_list);
    REQUIRE(output_lists[0].numAllocated == 1);
    REQUIRE(output_lists[1].numAllocated == 1);
    REQUIRE(output_lists[0].streamID == frames[0].streamID);
    REQUIRE(output_lists[1].streamID == frames[1].streamID);
    REQUIRE(output_lists[0].numFilled == 0);
    REQUIRE(output_lists[1].numFilled == 0);
    REQUIRE(std::memcmp(detections, input_copy, sizeof(detections)) == 0);

    output_lists[0].valid = true;
    output_lists[0].numFilled = 1;
    output_batch.numAllocated = 1;
    REQUIRE(context.processFrame(&params, &output_batch) == NvMOTStatus_Error);
    REQUIRE_FALSE(output_lists[0].valid);
    REQUIRE(output_lists[0].numFilled == 0);
    REQUIRE(context.removeStream(0) == NvMOTStatus_OK);
}

TEST_CASE("MixformerV2TRT keeps stable bbox on identical frames")
{
    const char *env_path = std::getenv("TRACK_TEST_CONFIG");
    std::string config_path =
        env_path != nullptr ? std::string(env_path)
                            : std::string(TRACK_TEST_CONFIG_PATH);

    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    REQUIRE(fs.isOpened());

    std::string image_path;
    std::string engine_path;
    fs["mixformerv2_engine"] >> engine_path;
    fs["image_path"] >> image_path;

    REQUIRE(!image_path.empty());
    REQUIRE(!engine_path.empty());

    cv::FileNode bbox_node = fs["init_bbox"];
    REQUIRE(bbox_node.isSeq());
    REQUIRE(bbox_node.size() == 4);

    float x0 = 0.f;
    float y0 = 0.f;
    float x1 = 0.f;
    float y1 = 0.f;
    bbox_node[0] >> x0;
    bbox_node[1] >> y0;
    bbox_node[2] >> x1;
    bbox_node[3] >> y1;

    int loop_count = 10;
    float max_abs_error = 1.0f;
    int template_size = 0;
    int search_size = 0;
    float template_factor = 0.f;
    float search_factor = 0.f;

    if (!fs["loop_count"].empty())
    {
        fs["loop_count"] >> loop_count;
    }
    if (!fs["max_abs_error"].empty())
    {
        fs["max_abs_error"] >> max_abs_error;
    }
    if (!fs["mixformerv2_template_size"].empty())
    {
        fs["mixformerv2_template_size"] >> template_size;
    }
    if (!fs["mixformerv2_search_size"].empty())
    {
        fs["mixformerv2_search_size"] >> search_size;
    }
    if (!fs["mixformerv2_template_factor"].empty())
    {
        fs["mixformerv2_template_factor"] >> template_factor;
    }
    if (!fs["mixformerv2_search_factor"].empty())
    {
        fs["mixformerv2_search_factor"] >> search_factor;
    }

    const std::string base_dir = dirName(config_path);
    image_path = resolvePath(base_dir, image_path);
    engine_path = resolvePath(base_dir, engine_path);

    if (!fileExists(engine_path))
    {
        WARN("MixformerV2 engine is unavailable; skipping optional model test: "
             << engine_path);
        return;
    }

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    REQUIRE(!img.empty());

    MixformerV2TRT tracker(engine_path);
    if (template_size > 0)
    {
        tracker.setTemplateSize(template_size);
    }
    if (search_size > 0)
    {
        tracker.setSearchSize(search_size);
    }
    if (template_factor > 0.f)
    {
        tracker.setTemplateFactor(template_factor);
    }
    if (search_factor > 0.f)
    {
        tracker.setSearchFactor(search_factor);
    }

    DrOBB bbox;
    bbox.box.x0 = x0;
    bbox.box.y0 = y0;
    bbox.box.x1 = x1;
    bbox.box.y1 = y1;
    bbox.class_id = 0;

    REQUIRE(tracker.init(img, bbox) == 0);

    for (int i = 0; i < loop_count; ++i)
    {
        const DrOBB &tracked = tracker.track(img);
        REQUIRE(std::abs(tracked.box.x0 - x0) <= max_abs_error);
        REQUIRE(std::abs(tracked.box.y0 - y0) <= max_abs_error);
        REQUIRE(std::abs(tracked.box.x1 - x1) <= max_abs_error);
        REQUIRE(std::abs(tracked.box.y1 - y1) <= max_abs_error);
    }
}
