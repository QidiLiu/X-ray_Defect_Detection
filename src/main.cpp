#include "absl/log/log.h"

#include "third_party/tensorrt-cpp-api-6.0/src/cmd_line_parser.h"
#include "third_party/tensorrt-cpp-api-6.0/src/engine.h"

int main(int argc, char *argv[]) {
    CommandLineArguments arguments;

    // Parse the command line arguments
    if (!parseArguments(argc, argv, arguments)) {
        return -1;
    }

    // Specify our GPU inference configuration options
    Options options;
    // Specify what precision to use for inference
    // FP16 is approximately twice as fast as FP32.
    options.precision = Precision::FP32;
    // If using INT8 precision, must specify path to directory containing
    // calibration data.
    options.calibrationDataDirectoryPath = "";
    // Specify the batch size to optimize for.
    options.optBatchSize = 1;
    // Specify the maximum batch size we plan on running.
    options.maxBatchSize = 1;

    Engine<float> engine(options);

    // Define our preprocessing code
    // The default Engine::build method will normalize values between [0.f, 1.f]
    // Setting the normalize flag to false will leave values between [0.f, 255.f]
    // (some converted models may require this).

    // For our YoloV8 model, we need the values to be normalized between
    // [0.f, 1.f] so we use the following params
    std::array<float, 3> subVals{0.f, 0.f, 0.f};
    std::array<float, 3> divVals{1.f, 1.f, 1.f};
    bool normalize = true;
    // Note, we could have also used the default values.

    // If the model requires values to be normalized between [-1.f, 1.f], use the
    // following params:
    //    subVals = {0.5f, 0.5f, 0.5f};
    //    divVals = {0.5f, 0.5f, 0.5f};
    //    normalize = true;

    // Load the TensorRT engine file directly
    bool succ = engine.loadNetwork(arguments.trtModelPath, subVals, divVals, normalize);
    if (!succ) {
        throw std::runtime_error("Unable to load TensorRT engine.");
    }

    // Load image and anchor into GPU
    const std::string in_img_filepath = "C:/Users/ben/Dev/X-ray_Defect_Detection/_data/train_crop_data/img_crop/10-20-11__NULL_1_2_sangdun-battery_separator_shadow_interference.png";
    const std::string in_anchor_filepath = "C:/Users/ben/Dev/X-ray_Defect_Detection/_data/train_crop_data/img_crop/10-20-11__NULL_1_2_sangdun-battery_separator_shadow_interference.png";
    auto cpu_img = cv::imread(in_img_filepath);
    if (cpu_img.empty()) { throw std::runtime_error("Unable to read image at path: " + in_img_filepath); }
    auto cpu_anchor = cv::imread(in_anchor_filepath);
    if (cpu_anchor.empty()) { throw std::runtime_error("Unable to read image at path: " + in_anchor_filepath); }
    cv::cuda::GpuMat img;
    img.upload(cpu_img);
    cv::cuda::GpuMat anchor;
    anchor.upload(cpu_anchor);

    // Preprocess the image and anchor
    cv::cuda::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::cuda::cvtColor(anchor, anchor, cv::COLOR_BGR2RGB);
    std::vector<std::vector<cv::cuda::GpuMat>> inputs;
    const auto &in_dims = engine.getInputDims();
    LOG(INFO) << "Input dimensions of in_img: " << in_dims[0].d[1] << "x" << in_dims[0].d[2];
    LOG(INFO) << "Input dimensions of in_anchor: " << in_dims[1].d[1] << "x" << in_dims[1].d[2];
    size_t batch_size = options.optBatchSize;

    std::vector<cv::cuda::GpuMat> in_img;
    for (size_t j = 0; j < batch_size; ++j) { // For each element we want to add to the batch...
        auto resized_img = Engine<float>::resizeKeepAspectRatioPadRightBottom(img, in_dims[0].d[1], in_dims[0].d[2]);
        in_img.emplace_back(std::move(resized_img));
    }
    inputs.emplace_back(std::move(in_img));
    std::vector<cv::cuda::GpuMat> in_anchor;
    for (size_t j = 0; j < batch_size; ++j) { // For each element we want to add to the batch...
        auto resized_img = Engine<float>::resizeKeepAspectRatioPadRightBottom(anchor, in_dims[1].d[1], in_dims[1].d[2]);
        in_anchor.emplace_back(std::move(resized_img));
    }
    inputs.emplace_back(std::move(in_anchor));
    
    // Warm up the network before we begin the benchmark
    std::cout << "\nWarming up the network..." << std::endl;
    std::vector<std::vector<std::vector<float>>> featureVectors;
    for (int i = 0; i < 100; ++i) {
        bool succ = engine.runInference(inputs, featureVectors);
        if (!succ) {
            throw std::runtime_error("Unable to run inference.");
        }
    }
    
    // Benchmark the inference time
    size_t numIterations = 1000;
    LOG(INFO) << "Warmup done. Running benchmarks (" << numIterations << " iterations)...\n";
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < numIterations; ++i) {
        featureVectors.clear();
        engine.runInference(inputs, featureVectors);
    }
    auto totalElapsedTimeMs = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
    auto avgElapsedTimeMs = totalElapsedTimeMs / numIterations / static_cast<float>(inputs[0].size());

    LOG(INFO) << "Benchmarking complete!";
    LOG(INFO) << "======================";
    LOG(INFO) << "Avg time per sample: ";
    LOG(INFO) << avgElapsedTimeMs << " ms";
    LOG(INFO) << "Batch size: ";
    LOG(INFO) << inputs[0].size();
    LOG(INFO) << "Avg FPS: ";
    LOG(INFO) << static_cast<int>(1000 / avgElapsedTimeMs) << " fps";
    LOG(INFO) << "======================\n";

    cv::Mat out_mask = cv::Mat::zeros(704, 352, CV_32FC1);
    for (size_t i = 0; i < 704; i++) {
        for (size_t j = 0; j < 352; j++) {
            out_mask.at<float>(i, j) = featureVectors[0][0][i * 352 + j];
        }
    }

    // Sigmoid activation
    cv::Mat exp_neg;
    cv::exp(-out_mask, exp_neg);
    out_mask = 1.0 / (1.0 + exp_neg);

    // Post-process the output mask
    const float threshold = 0.002;
    for (size_t i = 0; i < 704; i++) {
        for (size_t j = 0; j < 352; j++) {
            if (out_mask.at<float>(i, j) > threshold) {
                out_mask.at<float>(i, j) = 1.0;
            } else {
                out_mask.at<float>(i, j) = 0.0;
            }
        }
    }

    cv::Mat show_out_mask = cv::Mat::zeros(704, 352, CV_8UC1);
    out_mask.convertTo(show_out_mask, CV_8UC1, 255.0);
    cv::imshow("out_mask", show_out_mask);
    cv::waitKey(0);
    cv::destroyAllWindows();

    LOG(INFO) << "Wow, it works!";
}

