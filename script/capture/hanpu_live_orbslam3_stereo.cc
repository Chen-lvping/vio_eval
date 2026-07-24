#include <cstdio>
#include <csignal>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "System.h"

namespace {

volatile std::sig_atomic_t g_keep_running = 1;

void handle_signal(int) {
    g_keep_running = 0;
}

struct Options {
    std::string vocabulary_path;
    std::string settings_path;
    std::string video_device = "/dev/video0";
    std::string trajectory_stem = "hanpu_live";
    int width = 3840;
    int height = 1080;
    int fps = 30;
    bool crop_top_only = true;
    bool show_preview = true;
    bool enable_viewer = true;
    bool use_ffmpeg = true;
    bool use_clahe = true;
    double clahe_clip_limit = 3.0;
    int clahe_tile_grid = 8;
};

void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " <vocab> <settings> [options]\n"
        << "Options:\n"
        << "  --video-device <path>    Default /dev/video0\n"
        << "  --width <int>            Default 3840\n"
        << "  --height <int>           Default 1080\n"
        << "  --fps <int>              Default 30\n"
        << "  --trajectory-stem <str>  Default hanpu_live\n"
        << "  --no-top-crop            Use full decoded frame\n"
        << "  --no-preview             Disable OpenCV preview window\n"
        << "  --no-viewer              Disable Pangolin viewer\n"
        << "  --opencv-capture         Use OpenCV capture instead of ffmpeg pipe\n"
        << "  --no-clahe               Disable contrast enhancement\n";
}

bool parse_args(int argc, char** argv, Options& options) {
    if (argc < 3) {
        print_usage(argv[0]);
        return false;
    }

    options.vocabulary_path = argv[1];
    options.settings_path = argv[2];

    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--video-device") {
            options.video_device = require_value("--video-device");
        } else if (arg == "--width") {
            options.width = std::stoi(require_value("--width"));
        } else if (arg == "--height") {
            options.height = std::stoi(require_value("--height"));
        } else if (arg == "--fps") {
            options.fps = std::stoi(require_value("--fps"));
        } else if (arg == "--trajectory-stem") {
            options.trajectory_stem = require_value("--trajectory-stem");
        } else if (arg == "--no-top-crop") {
            options.crop_top_only = false;
        } else if (arg == "--no-preview") {
            options.show_preview = false;
        } else if (arg == "--no-viewer") {
            options.enable_viewer = false;
        } else if (arg == "--opencv-capture") {
            options.use_ffmpeg = false;
        } else if (arg == "--no-clahe") {
            options.use_clahe = false;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    return true;
}

class RawVideoPipe {
public:
    RawVideoPipe(const Options& options, int frame_width, int frame_height)
        : frame_width_(frame_width), frame_height_(frame_height) {
        std::ostringstream command;
        command
            << "ffmpeg -hide_banner -loglevel error "
            << "-fflags nobuffer "
            << "-flags low_delay "
            << "-thread_queue_size 2 "
            << "-analyzeduration 0 "
            << "-probesize 32 "
            << "-f v4l2 "
            << "-input_format mjpeg "
            << "-video_size " << options.width << "x" << options.height << " "
            << "-framerate " << options.fps << " "
            << "-i " << options.video_device << " "
            << "-threads 1 "
            << "-f rawvideo -pix_fmt bgr24 -";
        command_ = command.str();
        pipe_ = popen(command_.c_str(), "r");
        if (!pipe_) {
            throw std::runtime_error("failed to start ffmpeg capture pipeline");
        }
        buffer_.resize(static_cast<std::size_t>(frame_width_) * static_cast<std::size_t>(frame_height_) * 3U);
    }

    ~RawVideoPipe() {
        if (pipe_) {
            pclose(pipe_);
            pipe_ = nullptr;
        }
    }

    bool read(cv::Mat& frame_out) {
        if (!pipe_) {
            return false;
        }
        const std::size_t need = buffer_.size();
        const std::size_t got = fread(buffer_.data(), 1, need, pipe_);
        if (got != need) {
            return false;
        }
        cv::Mat wrapped(frame_height_, frame_width_, CV_8UC3, buffer_.data());
        frame_out = wrapped.clone();
        return true;
    }

private:
    std::string command_;
    FILE* pipe_ = nullptr;
    int frame_width_ = 0;
    int frame_height_ = 0;
    std::vector<unsigned char> buffer_;
};

bool open_with_opencv(const Options& options, cv::VideoCapture& cap) {
    if (options.video_device == "/dev/video0") {
        cap.open(0, cv::CAP_V4L2);
    } else {
        cap.open(options.video_device, cv::CAP_V4L2);
    }
    if (!cap.isOpened()) {
        return false;
    }
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, options.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, options.height);
    cap.set(cv::CAP_PROP_FPS, options.fps);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options;
        if (!parse_args(argc, argv, options)) {
            return 1;
        }

        std::signal(SIGINT, handle_signal);

        cv::Mat frame;
        std::unique_ptr<RawVideoPipe> ffmpeg_pipe;
        cv::VideoCapture cap;

        if (options.use_ffmpeg) {
            ffmpeg_pipe = std::make_unique<RawVideoPipe>(options, options.width, options.height);
            if (!ffmpeg_pipe->read(frame) || frame.empty()) {
                std::cerr << "failed to read the first frame from ffmpeg pipe for " << options.video_device << std::endl;
                return 3;
            }
        } else {
            if (!open_with_opencv(options, cap)) {
                std::cerr << "failed to open video device: " << options.video_device << std::endl;
                return 2;
            }
            if (!cap.read(frame) || frame.empty()) {
                std::cerr << "failed to read the first frame from " << options.video_device << std::endl;
                return 3;
            }
        }

        int full_width = frame.cols;
        int full_height = frame.rows;
        int cropped_height = options.crop_top_only ? (full_height / 2) : full_height;
        int single_width = full_width / 2;
        if (single_width <= 0 || cropped_height <= 0) {
            std::cerr << "unexpected frame size: " << full_width << "x" << full_height << std::endl;
            return 4;
        }

        std::cout << "video frame size: " << full_width << "x" << full_height
                  << " -> left/right " << single_width << "x" << cropped_height
                  << (options.crop_top_only ? " (top crop enabled)" : "") << std::endl;

        ORB_SLAM3::System slam(
            options.vocabulary_path,
            options.settings_path,
            ORB_SLAM3::System::STEREO,
            options.enable_viewer,
            0,
            options.trajectory_stem);

        const float image_scale = slam.GetImageScale();
        const auto start_time = std::chrono::steady_clock::now();
        std::size_t frame_count = 0;

        if (options.show_preview) {
            cv::namedWindow("hanpu_live_orbslam3", cv::WINDOW_NORMAL);
        }

        cv::Ptr<cv::CLAHE> clahe;
        if (options.use_clahe) {
            clahe = cv::createCLAHE(options.clahe_clip_limit, cv::Size(options.clahe_tile_grid, options.clahe_tile_grid));
        }

        while (g_keep_running) {
            const bool ok = options.use_ffmpeg ? ffmpeg_pipe->read(frame) : cap.read(frame);
            if (!ok || frame.empty()) {
                std::cerr << "warning: empty frame from camera, stopping" << std::endl;
                break;
            }

            cv::Mat working = options.crop_top_only ? frame(cv::Rect(0, 0, frame.cols, frame.rows / 2)).clone() : frame;
            cv::Mat left = working(cv::Rect(0, 0, working.cols / 2, working.rows));
            cv::Mat right = working(cv::Rect(working.cols / 2, 0, working.cols / 2, working.rows));

            cv::Mat left_gray;
            cv::Mat right_gray;
            cv::cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);
            if (clahe) {
                clahe->apply(left_gray, left_gray);
                clahe->apply(right_gray, right_gray);
            }

            const auto now = std::chrono::steady_clock::now();
            const double timestamp = std::chrono::duration<double>(now - start_time).count();

            if (image_scale != 1.0f) {
                cv::resize(left_gray, left_gray, cv::Size(), image_scale, image_scale);
                cv::resize(right_gray, right_gray, cv::Size(), image_scale, image_scale);
            }

            slam.TrackStereo(left_gray, right_gray, timestamp);
            ++frame_count;

            if (options.show_preview) {
                cv::Mat preview;
                cv::hconcat(left, right, preview);
                cv::putText(
                    preview,
                    "Press q or Ctrl-C to stop",
                    cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    cv::Scalar(0, 255, 0),
                    2,
                    cv::LINE_AA);
                cv::putText(
                    preview,
                    "Frames: " + std::to_string(frame_count),
                    cv::Point(20, 80),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    cv::Scalar(0, 255, 0),
                    2,
                    cv::LINE_AA);
                cv::imshow("hanpu_live_orbslam3", preview);
                const int key = cv::waitKey(1);
                if (key == 27 || key == 'q' || key == 'Q') {
                    break;
                }
            }
        }

        std::cout << "shutting down ORB-SLAM3 after " << frame_count << " frames" << std::endl;
        slam.Shutdown();

        if (!options.trajectory_stem.empty()) {
            slam.SaveTrajectoryEuRoC("f_" + options.trajectory_stem + ".txt");
            slam.SaveKeyFrameTrajectoryEuRoC("kf_" + options.trajectory_stem + ".txt");
        }

        if (cap.isOpened()) {
            cap.release();
        }
        if (options.show_preview) {
            cv::destroyAllWindows();
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "fatal: " << ex.what() << std::endl;
        return 10;
    }
}
