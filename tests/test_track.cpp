#include <catch2/catch_all.hpp>

#include "nanotrack_trt.h"
#include "mixformerv2_trt.h"

#include <opencv2/opencv.hpp>

#include <cmath>
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
