#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdio>
#include <algorithm>
#include <cctype>
#include <sys/stat.h>
#include <fstream>
#include <sstream>

#include "nanotrack_trt.h"
#include "mixformerv2_trt.h"

// 跟踪器基类抽象接口
class TrackerInterface {
public:
    virtual ~TrackerInterface() = default;
    virtual int init(const cv::Mat &img, DrOBB bbox) = 0;
    virtual const DrOBB &track(const cv::Mat &img) = 0;
};

// NanoTrack 包装器
class NanoTrackWrapper : public TrackerInterface {
private:
    NanotrackTRT tracker;
public:
    explicit NanoTrackWrapper(const std::string &merge_engine)
        : tracker(merge_engine) {}

    NanoTrackWrapper(const std::string &head_engine,
                     const std::string &backbone_engine,
                     const std::string &search_backbone_engine = "")
        : tracker(head_engine, backbone_engine, search_backbone_engine) {}
    
    int init(const cv::Mat &img, DrOBB bbox) override {
        return tracker.init(img, bbox);
    }
    
    const DrOBB &track(const cv::Mat &img) override {
        return tracker.track(img);
    }

    void setExemplarSize(int size) { tracker.setExemplarSize(size); }
    void setInstanceSize(int size) { tracker.setInstanceSize(size); }
};

// MixformerV2 包装器
class MixformerV2Wrapper : public TrackerInterface {
private:
    MixformerV2TRT tracker;
public:
    explicit MixformerV2Wrapper(const std::string &engine_path)
        : tracker(engine_path) {}
    
    int init(const cv::Mat &img, DrOBB bbox) override {
        return tracker.init(img, bbox);
    }
    
    const DrOBB &track(const cv::Mat &img) override {
        return tracker.track(img);
    }

    void setTemplateSize(int size) { tracker.setTemplateSize(size); }
    void setSearchSize(int size) { tracker.setSearchSize(size); }
    void setTemplateFactor(float f) { tracker.setTemplateFactor(f); }
    void setSearchFactor(float f) { tracker.setSearchFactor(f); }
    void setUpdateInterval(int interval) { tracker.setUpdateInterval(interval); }
    void setMaxScoreDecay(float decay) { tracker.setMaxScoreDecay(decay); }
    void setTemplateUpdateScoreThreshold(float threshold)
    {
        tracker.setTemplateUpdateScoreThreshold(threshold);
    }
};

struct TrackingResult {
    DrOBB bbox;
    float confidence;
    int frame_number;
};

#ifndef TRACK_VIDEO_TEST_CONFIG_PATH
#define TRACK_VIDEO_TEST_CONFIG_PATH "tests/track_video_test_config.yml"
#endif

bool fileExists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

bool directoryExists(const std::string& path) {
    struct stat st;
    return (stat(path.c_str(), &st) == 0);
}

bool createDirectory(const std::string& path) {
    std::string cmd = "mkdir -p " + path;
    return (system(cmd.c_str()) == 0);
}

std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    return ss.str();
}

int main(int argc, char* argv[]) {
    std::string config_path = TRACK_VIDEO_TEST_CONFIG_PATH;

    if (argc > 1) {
        config_path = argv[1];
    }

    if (!fileExists(config_path)) {
        std::cerr << "错误: 配置文件不存在: " << config_path << std::endl;
        return -1;
    }

    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "错误: 无法打开配置文件: " << config_path << std::endl;
        return -1;
    }

    std::string video_path;
    std::string nanotrack_mode = "merge";
    std::string nanotrack_merge_engine;
    std::string nanotrack_head_engine;
    std::string nanotrack_backbone_engine;
    std::string nanotrack_search_backbone_engine;
    std::string mixformerv2_engine;
    std::string mixformerv2_engine_small;
    std::string mixformerv2_engine_base;
    std::string output_dir;
    bool show_preview = false;
    float confidence_threshold = 0.3f;

    fs["video_path"] >> video_path;
    if (!fs["nanotrack_mode"].empty()) {
        fs["nanotrack_mode"] >> nanotrack_mode;
    }
    fs["nanotrack_merge_engine"] >> nanotrack_merge_engine;
    fs["nanotrack_head_engine"] >> nanotrack_head_engine;
    fs["nanotrack_backbone_engine"] >> nanotrack_backbone_engine;
    fs["nanotrack_search_backbone_engine"] >> nanotrack_search_backbone_engine;
    if (!fs["mixformerv2_engine"].empty()) {
        fs["mixformerv2_engine"] >> mixformerv2_engine;
    }
    if (!fs["mixformerv2_engine_small"].empty()) {
        fs["mixformerv2_engine_small"] >> mixformerv2_engine_small;
    }
    if (!fs["mixformerv2_engine_base"].empty()) {
        fs["mixformerv2_engine_base"] >> mixformerv2_engine_base;
    }
    fs["output_dir"] >> output_dir;
    fs["show_preview"] >> show_preview;

    if (!fs["confidence_threshold"].empty()) {
        fs["confidence_threshold"] >> confidence_threshold;
    }

    if (!fileExists(video_path)) {
        std::cerr << "错误: 视频文件不存在: " << video_path << std::endl;
        return -1;
    }

    std::cout << "=== 视频跟踪测试 ===" << std::endl;
    std::cout << "视频路径: " << video_path << std::endl;
    std::cout << "配置路径: " << config_path << std::endl;

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "错误: 无法打开视频文件: " << video_path << std::endl;
        return -1;
    }

    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::cout << "视频信息:" << std::endl;
    std::cout << "  总帧数: " << total_frames << std::endl;
    std::cout << "  帧率: " << fps << " FPS" << std::endl;
    std::cout << "  分辨率: " << width << "x" << height << std::endl;

    cv::FileNode bbox_node = fs["init_bbox"];
    if (bbox_node.empty() || !bbox_node.isSeq() || bbox_node.size() != 4) {
        std::cerr << "错误: init_bbox 必须包含4个值 [x0, y0, x1, y1]" << std::endl;
        return -1;
    }

    float x0 = 0.f, y0 = 0.f, x1 = 0.f, y1 = 0.f;
    bbox_node[0] >> x0;
    bbox_node[1] >> y0;
    bbox_node[2] >> x1;
    bbox_node[3] >> y1;

    std::cout << "初始边界框: [" << x0 << ", " << y0 << ", " << x1 << ", " << y1 << "]" << std::endl;
    std::cout << "置信度阈值: " << confidence_threshold << std::endl;

    int nanotrack_exemplar_size = 0;
    int nanotrack_instance_size = 0;
    if (!fs["nanotrack_exemplar_size"].empty()) {
        fs["nanotrack_exemplar_size"] >> nanotrack_exemplar_size;
    }
    if (!fs["nanotrack_instance_size"].empty()) {
        fs["nanotrack_instance_size"] >> nanotrack_instance_size;
    }
    int mixformerv2_template_size = 0;
    int mixformerv2_search_size = 0;
    float mixformerv2_template_factor = 0.f;
    float mixformerv2_search_factor = 0.f;
    int mixformerv2_update_interval = -1;
    float mixformerv2_max_score_decay = -1.f;
    float mixformerv2_template_update_score_threshold = -1.f;
    if (!fs["mixformerv2_template_size"].empty()) {
        fs["mixformerv2_template_size"] >> mixformerv2_template_size;
    }
    if (!fs["mixformerv2_search_size"].empty()) {
        fs["mixformerv2_search_size"] >> mixformerv2_search_size;
    }
    if (!fs["mixformerv2_template_factor"].empty()) {
        fs["mixformerv2_template_factor"] >> mixformerv2_template_factor;
    }
    if (!fs["mixformerv2_search_factor"].empty()) {
        fs["mixformerv2_search_factor"] >> mixformerv2_search_factor;
    }
    if (!fs["mixformerv2_update_interval"].empty()) {
        fs["mixformerv2_update_interval"] >> mixformerv2_update_interval;
    }
    if (!fs["mixformerv2_max_score_decay"].empty()) {
        fs["mixformerv2_max_score_decay"] >> mixformerv2_max_score_decay;
    }
    if (!fs["mixformerv2_template_update_score_threshold"].empty()) {
        fs["mixformerv2_template_update_score_threshold"] >>
            mixformerv2_template_update_score_threshold;
    }

    std::string timestamp = getCurrentTimestamp();
    if (output_dir.empty()) {
        output_dir = "tests/output_videos_" + timestamp;
    }

    if (!directoryExists(output_dir)) {
        if (!createDirectory(output_dir)) {
            std::cerr << "错误: 无法创建输出目录: " << output_dir << std::endl;
            return -1;
        }
    }

    std::string output_video_path = output_dir + "/tracking_result_" + timestamp + ".mp4";
    std::string output_info_path = output_dir + "/tracking_info_" + timestamp + ".txt";

    cv::VideoWriter writer;
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    writer.open(output_video_path, fourcc, fps, cv::Size(width, height));

    if (!writer.isOpened()) {
        std::cerr << "错误: 无法创建输出视频文件: " << output_video_path << std::endl;
        return -1;
    }

    std::cout << "输出目录: " << output_dir << std::endl;
    std::cout << "输出视频: " << output_video_path << std::endl;

    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    cv::Mat first_frame;
    cap >> first_frame;

    if (first_frame.empty()) {
        std::cerr << "错误: 无法读取第一帧" << std::endl;
        return -1;
    }

    std::cout << "\n初始化跟踪器..." << std::endl;
    
    // 创建跟踪器
    std::vector<std::pair<std::string, TrackerInterface*>> trackers;
    
    if (nanotrack_mode != "merge" && nanotrack_mode != "split") {
        std::cerr << "错误: nanotrack_mode 仅支持 merge 或 split" << std::endl;
        return -1;
    }

    // NanoTrack 跟踪器
    if (nanotrack_mode == "merge") {
        if (nanotrack_merge_engine.empty()) {
            std::cerr << "错误: nanotrack_merge_engine 为空" << std::endl;
            return -1;
        }
        auto nano_tracker = new NanoTrackWrapper(nanotrack_merge_engine);
        if (nanotrack_exemplar_size > 0) {
            nano_tracker->setExemplarSize(nanotrack_exemplar_size);
        }
        if (nanotrack_instance_size > 0) {
            nano_tracker->setInstanceSize(nanotrack_instance_size);
        }
        trackers.push_back({"NanoTrack", nano_tracker});
    } else {
        if (nanotrack_head_engine.empty() || nanotrack_backbone_engine.empty()) {
            std::cerr << "错误: nanotrack_head_engine 或 nanotrack_backbone_engine 为空" << std::endl;
            return -1;
        }
        auto nano_tracker = new NanoTrackWrapper(nanotrack_head_engine,
                                                  nanotrack_backbone_engine,
                                                  nanotrack_search_backbone_engine);
        if (nanotrack_exemplar_size > 0) {
            nano_tracker->setExemplarSize(nanotrack_exemplar_size);
        }
        if (nanotrack_instance_size > 0) {
            nano_tracker->setInstanceSize(nanotrack_instance_size);
        }
        trackers.push_back({"NanoTrack", nano_tracker});
    }
    
    // MixformerV2 跟踪器
    auto add_mixformerv2_tracker = [&](const std::string &name,
                                       const std::string &engine_path) {
        if (engine_path.empty()) {
            return;
        }
        auto mixformer_tracker = new MixformerV2Wrapper(engine_path);
        if (mixformerv2_template_size > 0) {
            mixformer_tracker->setTemplateSize(mixformerv2_template_size);
        }
        if (mixformerv2_search_size > 0) {
            mixformer_tracker->setSearchSize(mixformerv2_search_size);
        }
        if (mixformerv2_template_factor > 0.f) {
            mixformer_tracker->setTemplateFactor(mixformerv2_template_factor);
        }
        if (mixformerv2_search_factor > 0.f) {
            mixformer_tracker->setSearchFactor(mixformerv2_search_factor);
        }
        if (mixformerv2_update_interval > 0) {
            mixformer_tracker->setUpdateInterval(mixformerv2_update_interval);
        }
        if (mixformerv2_max_score_decay >= 0.f) {
            mixformer_tracker->setMaxScoreDecay(mixformerv2_max_score_decay);
        }
        if (mixformerv2_template_update_score_threshold >= 0.f) {
            mixformer_tracker->setTemplateUpdateScoreThreshold(
                mixformerv2_template_update_score_threshold);
        }
        trackers.push_back({name, mixformer_tracker});
    };

    add_mixformerv2_tracker("MixformerV2Base", mixformerv2_engine_base);
    add_mixformerv2_tracker("MixformerV2Small", mixformerv2_engine_small);
    if (mixformerv2_engine_base.empty() && mixformerv2_engine_small.empty()) {
        add_mixformerv2_tracker("MixformerV2", mixformerv2_engine);
    }
    
    if (trackers.empty()) {
        std::cerr << "错误: 未配置任何跟踪器引擎" << std::endl;
        return -1;
    }

    int nanotrack_index = -1;
    std::vector<int> mixformerv2_indexes;
    for (size_t i = 0; i < trackers.size(); ++i) {
        if (trackers[i].first == "NanoTrack") {
            nanotrack_index = static_cast<int>(i);
        } else if (trackers[i].first.find("MixformerV2") == 0) {
            mixformerv2_indexes.push_back(static_cast<int>(i));
        }
    }

    DrOBB init_bbox;
    init_bbox.box.x0 = x0;
    init_bbox.box.y0 = y0;
    init_bbox.box.x1 = x1;
    init_bbox.box.y1 = y1;
    init_bbox.class_id = 0;

    for (auto &tracker_pair : trackers) {
        if (tracker_pair.second->init(first_frame, init_bbox) != 0) {
            std::cerr << "错误: " << tracker_pair.first << " 初始化失败" << std::endl;
            return -1;
        }
        std::cout << tracker_pair.first << " 初始化成功!" << std::endl;
    }

    std::vector<cv::VideoWriter> per_tracker_writers(trackers.size());
    std::vector<std::string> per_tracker_video_paths(trackers.size());
    std::vector<bool> per_tracker_writer_open(trackers.size(), false);
    auto make_safe_name = [](std::string name) {
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);
        for (char &c : name) {
            if (!std::isalnum(static_cast<unsigned char>(c))) {
                c = '_';
            }
        }
        return name;
    };
    for (size_t i = 0; i < trackers.size(); ++i) {
        std::string safe_name = make_safe_name(trackers[i].first);
        per_tracker_video_paths[i] =
            output_dir + "/tracking_" + safe_name + "_" + timestamp + ".mp4";
        per_tracker_writers[i].open(per_tracker_video_paths[i], fourcc, fps, cv::Size(width, height));
        if (!per_tracker_writers[i].isOpened()) {
            std::cerr << "警告: 无法创建单独输出视频: " << per_tracker_video_paths[i] << std::endl;
            continue;
        }
        per_tracker_writer_open[i] = true;
        std::cout << "单独输出视频: " << per_tracker_video_paths[i] << std::endl;
    }

    std::ofstream info_file(output_info_path);
    if (!info_file.is_open()) {
        std::cerr << "警告: 无法创建信息文件: " << output_info_path << std::endl;
    } else {
        info_file << "视频跟踪测试信息\n";
        info_file << "==================\n\n";
        info_file << "视频路径: " << video_path << "\n";
        info_file << "总帧数: " << total_frames << "\n";
        info_file << "帧率: " << fps << " FPS\n";
        info_file << "分辨率: " << width << "x" << height << "\n";
        info_file << "初始边界框: [" << x0 << ", " << y0 << ", " << x1 << ", " << y1 << "]\n";
        info_file << "置信度阈值: " << confidence_threshold << "\n";
        info_file << "跟踪器: ";
        for (size_t i = 0; i < trackers.size(); ++i) {
            if (i > 0) info_file << ", ";
            info_file << trackers[i].first;
        }
        info_file << "\n\n";
        info_file << "跟踪结果:\n";
        info_file << "帧号";
        for (auto &tracker_pair : trackers) {
            info_file << ", " << tracker_pair.first << "_x0, " << tracker_pair.first << "_y0, "
                     << tracker_pair.first << "_x1, " << tracker_pair.first << "_y1, "
                     << tracker_pair.first << "_置信度";
        }
        info_file << "\n";
    }

    std::vector<TrackingResult> results;
    results.reserve(total_frames);

    // 记录 tracker.track 处理时间（整体与按模型分别统计）
    std::vector<double> track_times_ms;
    track_times_ms.reserve(total_frames);
    std::vector<double> per_tracker_total_ms(trackers.size(), 0.0);
    std::vector<size_t> per_tracker_count(trackers.size(), 0);
    std::vector<double> per_tracker_window_ms(trackers.size(), 0.0);
    std::vector<size_t> per_tracker_window_count(trackers.size(), 0);
    std::vector<double> per_tracker_window_fps(trackers.size(), 0.0);
    auto window_start_time = std::chrono::high_resolution_clock::now();
    const double window_seconds = 3.0;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int frame_idx = 0; frame_idx < total_frames; ++frame_idx) {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            std::cout << "警告: 第 " << frame_idx << " 帧为空" << std::endl;
            break;
        }

        // 对所有跟踪器进行跟踪
        std::vector<std::pair<std::string, DrOBB>> tracking_results;
        std::vector<double> per_model_fps;
        for (size_t ti = 0; ti < trackers.size(); ++ti) {
            auto &tracker_pair = trackers[ti];
            auto track_start = std::chrono::high_resolution_clock::now();
            const DrOBB& tracked = tracker_pair.second->track(frame);
            auto track_end = std::chrono::high_resolution_clock::now();
            auto track_duration = std::chrono::duration_cast<std::chrono::microseconds>(track_end - track_start);
            double track_time_ms = track_duration.count() / 1000.0;
            track_times_ms.push_back(track_time_ms);
            per_tracker_total_ms[ti] += track_time_ms;
            per_tracker_count[ti] += 1;
            per_tracker_window_ms[ti] += track_time_ms;
            per_tracker_window_count[ti] += 1;
            double track_fps_model = track_time_ms > 0.0 ? 1000.0 / track_time_ms : 0.0;
            per_model_fps.push_back(track_fps_model);

            tracking_results.push_back({tracker_pair.first, tracked});
        }

        auto window_now = std::chrono::high_resolution_clock::now();
        double window_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(window_now - window_start_time).count();
        if (window_elapsed >= window_seconds) {
            for (size_t i = 0; i < trackers.size(); ++i) {
                if (per_tracker_window_count[i] > 0) {
                    double avg_ms = per_tracker_window_ms[i] / static_cast<double>(per_tracker_window_count[i]);
                    per_tracker_window_fps[i] = avg_ms > 0.0 ? 1000.0 / avg_ms : 0.0;
                } else {
                    per_tracker_window_fps[i] = 0.0;
                }
                per_tracker_window_ms[i] = 0.0;
                per_tracker_window_count[i] = 0;
            }
            window_start_time = window_now;
        } else {
            for (size_t i = 0; i < trackers.size(); ++i) {
                if (per_tracker_window_count[i] > 0) {
                    double avg_ms = per_tracker_window_ms[i] / static_cast<double>(per_tracker_window_count[i]);
                    per_tracker_window_fps[i] = avg_ms > 0.0 ? 1000.0 / avg_ms : 0.0;
                } else {
                    per_tracker_window_fps[i] = 0.0;
                }
            }
        }

        // 获取第一个跟踪器的结果用于视频输出（可视化）
        const DrOBB& tracked = tracking_results[0].second;
        TrackingResult result;
        result.bbox = tracked;
        result.confidence = tracked.score;
        result.frame_number = frame_idx;

        results.push_back(result);

        cv::Mat output_frame = frame.clone();

        // 绘制所有跟踪器的结果
        const cv::Scalar colors[] = {
            cv::Scalar(0, 255, 0),    // 绿色：NanoTrack
            cv::Scalar(0, 0, 255),    // 红色：MixformerV2
            cv::Scalar(0, 255, 255)   // 黄色：其他
        };

        auto draw_tracker_overlay = [&](cv::Mat &img, size_t i, double font_scale_factor) {
            const DrOBB& tracked_result = tracking_results[i].second;
            cv::Scalar color = colors[i % 3];
            const double label_font_scale = 0.5 * font_scale_factor;
            int label_thickness = static_cast<int>(std::round(2 * font_scale_factor));
            if (label_thickness < 1) {
                label_thickness = 1;
            }

            // 绘制目标框
            cv::rectangle(img,
                         cv::Point(tracked_result.box.x0, tracked_result.box.y0),
                         cv::Point(tracked_result.box.x1, tracked_result.box.y1),
                         (tracked_result.score >= confidence_threshold ? color : cv::Scalar(0, 0, 255)), 2);

            // 文本内容
            const bool ok = tracked_result.score >= confidence_threshold;
            std::string label = ok
                ? cv::format("%s: %.3f", tracking_results[i].first.c_str(), tracked_result.score)
                : cv::format("%s Lost (%.3f)", tracking_results[i].first.c_str(), tracked_result.score);

            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, label_font_scale, label_thickness, &baseline);

            // 交替上下放置文字，避免互相遮挡
            const int margin = std::max(2, static_cast<int>(std::round(5 * font_scale_factor)));
            int text_x0 = std::max(0, static_cast<int>(tracked_result.box.x0));
            int text_y0;
            std::string tracker_name_lower = tracking_results[i].first;
            std::transform(tracker_name_lower.begin(), tracker_name_lower.end(), tracker_name_lower.begin(), ::tolower);
            const bool place_left = tracker_name_lower.find("mixformerv2small") != std::string::npos;
            if (place_left) {
                const int anchor_x = static_cast<int>(tracked_result.box.x0);
                const int anchor_y = static_cast<int>(tracked_result.box.y0);
                text_x0 = anchor_x - text_size.width - margin;
                if (text_x0 < 0) {
                    text_x0 = 0;
                }
                text_y0 = anchor_y;
                if (text_y0 + text_size.height + baseline > output_frame.rows) {
                    text_y0 = std::max(0, output_frame.rows - text_size.height - baseline);
                } else if (text_y0 < 0) {
                    text_y0 = 0;
                }
            } else if (i % 2 == 0) { // 顶部
                text_y0 = static_cast<int>(tracked_result.box.y0) - text_size.height - baseline - margin;
                text_y0 = std::max(text_y0, 0);
            } else {         // 底部
                text_y0 = static_cast<int>(tracked_result.box.y1) + margin;
                if (text_y0 + text_size.height + baseline > output_frame.rows) {
                    text_y0 = std::max(0, output_frame.rows - text_size.height - baseline);
                }
            }

            cv::Point rect_tl(text_x0, text_y0);
            cv::Point rect_br(text_x0 + text_size.width, text_y0 + text_size.height + baseline);
            if (ok) {
                cv::rectangle(img, rect_tl, rect_br, color, -1);
                cv::putText(img, label,
                           cv::Point(text_x0, text_y0 + text_size.height),
                           cv::FONT_HERSHEY_SIMPLEX, label_font_scale,
                           cv::Scalar(0, 0, 0), label_thickness);
            } else {
                cv::rectangle(img, rect_tl, rect_br, cv::Scalar(0, 0, 255), -1);
                cv::putText(img, label,
                           cv::Point(text_x0, text_y0 + text_size.height),
                           cv::FONT_HERSHEY_SIMPLEX, label_font_scale,
                           cv::Scalar(255, 255, 255), label_thickness);
            }
        };

        for (size_t i = 0; i < tracking_results.size(); ++i) {
            draw_tracker_overlay(output_frame, i, 1.0);
        }

        cv::putText(output_frame, cv::format("Frame: %d/%d", frame_idx + 1, total_frames),
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                   cv::Scalar(255, 255, 255), 2);

        if (nanotrack_index >= 0 || !mixformerv2_indexes.empty()) {
            const int padding = 6;
            const int margin = 10;
            std::vector<std::string> fps_lines;
            if (nanotrack_index >= 0) {
                double fps_value = per_tracker_window_fps[nanotrack_index];
                std::string fps_text = fps_value > 0.0
                    ? cv::format("nanotrack avg fps: %.1f", fps_value)
                    : "nanotrack avg fps: --";
                fps_lines.push_back(fps_text);
            }
            for (int idx : mixformerv2_indexes) {
                double fps_value = per_tracker_window_fps[idx];
                std::string name = trackers[idx].first;
                std::transform(name.begin(), name.end(), name.begin(), ::tolower);
                std::string fps_text = fps_value > 0.0
                    ? cv::format("%s avg fps: %.1f", name.c_str(), fps_value)
                    : cv::format("%s avg fps: --", name.c_str());
                fps_lines.push_back(fps_text);
            }

            int max_width = 0;
            int total_height = 0;
            std::vector<cv::Size> text_sizes;
            std::vector<int> baselines;
            for (const auto &line : fps_lines) {
                int baseline = 0;
                cv::Size text_size = cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
                text_sizes.push_back(text_size);
                baselines.push_back(baseline);
                max_width = std::max(max_width, text_size.width);
                total_height += text_size.height + baseline;
            }
            total_height += static_cast<int>(fps_lines.size() - 1) * 4;

            int box_x = std::max(0, output_frame.cols - max_width - padding * 2 - margin);
            int box_y = margin;
            int box_w = max_width + padding * 2;
            int box_h = total_height + padding * 2;
            if (box_x + box_w > output_frame.cols) {
                box_x = std::max(0, output_frame.cols - box_w);
            }

            cv::rectangle(output_frame,
                         cv::Point(box_x, box_y),
                         cv::Point(box_x + box_w, box_y + box_h),
                         cv::Scalar(0, 0, 0), -1);

            int text_y = box_y + padding;
            for (size_t i = 0; i < fps_lines.size(); ++i) {
                text_y += text_sizes[i].height;
                cv::putText(output_frame, fps_lines[i],
                           cv::Point(box_x + padding, text_y),
                           cv::FONT_HERSHEY_SIMPLEX, 0.6,
                           cv::Scalar(255, 255, 255), 2);
                text_y += baselines[i] + 4;
            }
        }

        writer.write(output_frame);
        for (size_t i = 0; i < tracking_results.size() && i < per_tracker_writers.size(); ++i) {
            if (!per_tracker_writer_open[i]) {
                continue;
            }
            cv::Mat single_frame = frame.clone();
            draw_tracker_overlay(single_frame, i, 2.0);
            cv::putText(single_frame, cv::format("Frame: %d/%d", frame_idx + 1, total_frames),
                       cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.4,
                       cv::Scalar(255, 255, 255), 4);
            double fps_value = per_tracker_window_fps[i];
            std::string fps_text = fps_value > 0.0
                ? cv::format("avg fps: %.1f", fps_value)
                : "avg fps: --";
            cv::putText(single_frame, fps_text,
                       cv::Point(10, 65), cv::FONT_HERSHEY_SIMPLEX, 1.2,
                       cv::Scalar(255, 255, 255), 4);
            per_tracker_writers[i].write(single_frame);
        }

        if (info_file.is_open()) {
            info_file << frame_idx;
            for (const auto &tracker_result : tracking_results) {
                info_file << ", "
                         << tracker_result.second.box.x0 << ", " << tracker_result.second.box.y0 << ", "
                         << tracker_result.second.box.x1 << ", " << tracker_result.second.box.y1 << ", "
                         << tracker_result.second.score;
            }
            info_file << "\n";
        }

        // 每一帧打印各模型 FPS（控制台单行动态刷新）
        {
            std::ostringstream fps_line;
            fps_line << " | FPS:";
            for (size_t i = 0; i < trackers.size() && i < per_model_fps.size(); ++i) {
                fps_line << " " << trackers[i].first << "="
                         << std::fixed << std::setprecision(1) << per_model_fps[i];
            }
            std::cout << "\r进度: " << frame_idx + 1 << "/" << total_frames
                      << " (" << std::fixed << std::setprecision(1)
                      << (100.0 * (frame_idx + 1) / total_frames) << "%)"
                      << " 置信度: " << std::setprecision(3) << tracked.score
                      << fps_line.str()
                      << "                " << std::flush;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // 计算 tracker.track 的平均处理时间和 FPS（整体）
    double total_track_time_ms = 0.0;
    for (double t : track_times_ms) {
        total_track_time_ms += t;
    }
    double avg_track_time_ms = track_times_ms.empty() ? 0.0 : total_track_time_ms / track_times_ms.size();
    double avg_track_fps = avg_track_time_ms > 0.0 ? 1000.0 / avg_track_time_ms : 0.0;

    std::cout << "\n\n=== 跟踪完成 ===" << std::endl;
    std::cout << "总处理时间: " << duration.count() / 1000.0 << " 秒" << std::endl;
    std::cout << "平均FPS: " << (total_frames * 1000.0) / duration.count() << std::endl;
    std::cout << "\nTracker.track 性能统计:" << std::endl;
    std::cout << "  平均处理时间: " << std::fixed << std::setprecision(2) << avg_track_time_ms << " ms" << std::endl;
    std::cout << "  平均 FPS: " << std::setprecision(1) << avg_track_fps << std::endl;
    // 各模型平均 FPS
    if (!trackers.empty()) {
        std::cout << "\n各模型平均 FPS:" << std::endl;
        for (size_t i = 0; i < trackers.size(); ++i) {
            double avg_ms = per_tracker_count[i] ? (per_tracker_total_ms[i] / static_cast<double>(per_tracker_count[i])) : 0.0;
            double avg_fps_model = avg_ms > 0.0 ? 1000.0 / avg_ms : 0.0;
            std::cout << "  - " << trackers[i].first << ": " << std::fixed << std::setprecision(1) << avg_fps_model << " FPS" << std::endl;
        }
    }
    std::cout << "\n使用的跟踪器: ";
    for (size_t i = 0; i < trackers.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << trackers[i].first;
    }
    std::cout << std::endl;

    if (info_file.is_open()) {
        info_file << "\n统计信息:\n";
        info_file << "总处理时间: " << duration.count() / 1000.0 << " 秒\n";
        info_file << "平均FPS: " << (total_frames * 1000.0) / duration.count() << "\n";
        info_file << "\nTracker.track 性能统计:\n";
        info_file << "平均处理时间: " << avg_track_time_ms << " ms\n";
        info_file << "平均 FPS: " << avg_track_fps << "\n";
        if (!trackers.empty()) {
            info_file << "各模型平均 FPS:\n";
            for (size_t i = 0; i < trackers.size(); ++i) {
                double avg_ms = per_tracker_count[i] ? (per_tracker_total_ms[i] / static_cast<double>(per_tracker_count[i])) : 0.0;
                double avg_fps_model = avg_ms > 0.0 ? 1000.0 / avg_ms : 0.0;
                info_file << trackers[i].first << ": " << avg_fps_model << " FPS\n";
            }
        }

        float avg_confidence = 0.0f;
        int valid_frames = 0;
        for (const auto& r : results) {
            if (r.confidence >= confidence_threshold) {
                avg_confidence += r.confidence;
                valid_frames++;
            }
        }
        if (valid_frames > 0) {
            avg_confidence /= valid_frames;
        }
        info_file << "平均置信度: " << avg_confidence << "\n";
        info_file << "有效跟踪帧数: " << valid_frames << "/" << total_frames << "\n";
        info_file.close();
    }

    cap.release();
    writer.release();
    for (auto &w : per_tracker_writers) {
        if (w.isOpened()) {
            w.release();
        }
    }

    // 清理跟踪器
    for (auto &tracker_pair : trackers) {
        delete tracker_pair.second;
    }
    trackers.clear();

    if (show_preview) {
        cv::destroyAllWindows();
    }

    std::cout << "结果已保存到: " << output_dir << std::endl;
    std::cout << "测试通过!" << std::endl;

    return 0;
}
