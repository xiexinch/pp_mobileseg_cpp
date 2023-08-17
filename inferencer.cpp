#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;

const vector<vector<int>> ade20k_palette = {{120, 120, 120}, {180, 120, 120}, {6, 230, 230}, {80, 50, 50},
                                            {4, 200, 3}, {120, 120, 80}, {140, 140, 140}, {204, 5, 255},
                                            {230, 230, 230}, {4, 250, 7}, {224, 5, 255}, {235, 255, 7},
                                            {150, 5, 61}, {120, 120, 70}, {8, 255, 51}, {255, 6, 82},
                                            {143, 255, 140}, {204, 255, 4}, {255, 51, 7}, {204, 70, 3},
                                            {0, 102, 200}, {61, 230, 250}, {255, 6, 51}, {11, 102, 255},
                                            {255, 7, 71}, {255, 9, 224}, {9, 7, 230}, {220, 220, 220},
                                            {255, 9, 92}, {112, 9, 255}, {8, 255, 214}, {7, 255, 224},
                                            {255, 184, 6}, {10, 255, 71}, {255, 41, 10}, {7, 255, 255},
                                            {224, 255, 8}, {102, 8, 255}, {255, 61, 6}, {255, 194, 7},
                                            {255, 122, 8}, {0, 255, 20}, {255, 8, 41}, {255, 5, 153},
                                            {6, 51, 255}, {235, 12, 255}, {160, 150, 20}, {0, 163, 255},
                                            {140, 140, 140}, {250, 10, 15}, {20, 255, 0}, {31, 255, 0},
                                            {255, 31, 0}, {255, 224, 0}, {153, 255, 0}, {0, 0, 255},
                                            {255, 71, 0}, {0, 235, 255}, {0, 173, 255}, {31, 0, 255},
                                            {11, 200, 200}, {255, 82, 0}, {0, 255, 245}, {0, 61, 255},
                                            {0, 255, 112}, {0, 255, 133}, {255, 0, 0}, {255, 163, 0},
                                            {255, 102, 0}, {194, 255, 0}, {0, 143, 255}, {51, 255, 0},
                                            {0, 82, 255}, {0, 255, 41}, {0, 255, 173}, {10, 0, 255},
                                            {173, 255, 0}, {0, 255, 153}, {255, 92, 0}, {255, 0, 255},
                                            {255, 0, 245}, {255, 0, 102}, {255, 173, 0}, {255, 0, 20},
                                            {255, 184, 184}, {0, 31, 255}, {0, 255, 61}, {0, 71, 255},
                                            {255, 0, 204}, {0, 255, 194}, {0, 255, 82}, {0, 10, 255},
                                            {0, 112, 255}, {51, 0, 255}, {0, 194, 255}, {0, 122, 255},
                                            {0, 255, 163}, {255, 153, 0}, {0, 255, 10}, {255, 112, 0},
                                            {143, 255, 0}, {82, 0, 255}, {163, 255, 0}, {255, 235, 0},
                                            {8, 184, 170}, {133, 0, 255}, {0, 255, 92}, {184, 0, 255},
                                            {255, 0, 31}, {0, 184, 255}, {0, 214, 255}, {255, 0, 112},
                                            {92, 255, 0}, {0, 224, 255}, {112, 224, 255}, {70, 184, 160},
                                            {163, 0, 255}, {153, 0, 255}, {71, 255, 0}, {255, 0, 163},
                                            {255, 204, 0}, {255, 0, 143}, {0, 255, 235}, {133, 255, 0},
                                            {255, 0, 235}, {245, 0, 255}, {255, 0, 122}, {255, 245, 0},
                                            {10, 190, 212}, {214, 255, 0}, {0, 204, 255}, {20, 0, 255},
                                            {255, 255, 0}, {0, 153, 255}, {0, 41, 255}, {0, 255, 204},
                                            {41, 0, 255}, {41, 255, 0}, {173, 0, 255}, {0, 245, 255},
                                            {71, 0, 255}, {122, 0, 255}, {0, 255, 184}, {0, 92, 255},
                                            {184, 255, 0}, {0, 133, 255}, {255, 214, 0}, {25, 194, 194},
                                            {102, 255, 0}, {92, 0, 255}};

class PPMobileSeg
{
public:
    PPMobileSeg();
    Mat inference(cv::Mat srcImage);

private:
    void preprocess(const cv::Mat& srcImage);
    int input_width;
    int input_height;
    std::vector<float> input_image_;

    Ort::Env env = Env(ORT_LOGGING_LEVEL_ERROR, "pp_mobielseg");
    Ort::Session *ort_session;
    Ort::SessionOptions session_options = Ort::SessionOptions();
    std::vector<const char *> inputNodeNames;
    std::vector<const char *> outputNodeNames;
    std::vector<AllocatedStringPtr> inputNodeNameAllocatedStrings;
    std::vector<AllocatedStringPtr> outputNodeNameAllocatedStrings;

    std::vector<std::vector<int64_t>> input_node_dims;  // >=1 outputs
    std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs

    std::vector<cv::Vec3b> colors;
    std::array <float, 3> _mean;
    std::array<float, 3> _std;
    const std::string model_path;
};

PPMobileSeg::PPMobileSeg() :
        model_path("../end2end.onnx"),
        ort_session(nullptr),
        input_width(0),
        input_height(0)
{
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    ort_session = new Ort::Session(env, model_path.c_str(), session_options);

    size_t num_input_nodes = ort_session->GetInputCount();
    size_t num_output_nodes = ort_session->GetOutputCount();
    Ort::AllocatorWithDefaultOptions allocator;

    for (size_t i = 0; i < num_input_nodes; ++i) {
        inputNodeNameAllocatedStrings.emplace_back(ort_session->GetInputNameAllocated(i, allocator));
        inputNodeNames.push_back(inputNodeNameAllocatedStrings.back().get());

        Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        input_node_dims.push_back(input_tensor_info.GetShape());
    }

    for (size_t i = 0; i < num_output_nodes; ++i) {
        outputNodeNameAllocatedStrings.emplace_back(ort_session->GetOutputNameAllocated(i, allocator));
        outputNodeNames.push_back(outputNodeNameAllocatedStrings.back().get());

        Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        output_node_dims.push_back(output_tensor_info.GetShape());
    }

    input_width = input_node_dims[0][3];
    input_height = input_node_dims[0][2];

    for (const auto& color : ade20k_palette) {
        colors.emplace_back(color[0], color[1], color[2]);
    }

    std::array<float, 3> mean = {123.675f, 116.28f, 103.53f};
    std::array<float, 3> std = {58.395f, 57.12f, 57.375f};
    for (size_t i = 0; i < 3; ++i) {
        _mean[i] = mean[i];
        _std[i] = 1.0f / std[i];
    }
}


void PPMobileSeg::preprocess(const cv::Mat& srcImage)
{
    cv::Mat dstimg;
    cv::resize(srcImage, dstimg, cv::Size(input_width, input_height));
    int row = dstimg.rows;
    int col = dstimg.cols;

    input_image_.resize(row * col * 3);

    cv::MatIterator_<cv::Vec3b> it, end;
    for (it = dstimg.begin<cv::Vec3b>(), end = dstimg.end<cv::Vec3b>(); it != end; ++it)
    {
        const cv::Vec3b &pix = *it;
        for (int c = 0; c < 3; c++)
        {
            input_image_[c * row * col + (it.pos().y * col ) + it.pos().x] = ((float)pix[c] - _mean[c]) * _std[c];
        }
    }

}

cv::Mat PPMobileSeg::inference(cv::Mat srcImage)
{
    this->preprocess(srcImage);
    std::array<int64_t, 4> input_shape = {1, 3, input_height, input_width};

    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        allocator_info, input_image_.data(), input_image_.size(), input_shape.data(), 4);

    std::vector<Value> ort_outputs = ort_session->Run(
        Ort::RunOptions{nullptr},
        inputNodeNames.data(),
        &input_tensor,
        1,
        outputNodeNames.data(),
        1);

    // post process
    Ort::Value &mask_pred = ort_outputs.at(0);
    const int out_h = (int)output_node_dims[0][2];
    const int out_w = (int)output_node_dims[0][3];
    auto *mask_prt = mask_pred.GetTensorMutableData<float>();
    cv::Mat mask_out(out_h, out_w, CV_32FC2, mask_prt);

    cv::Mat segmentation_map;
    resize(mask_out, segmentation_map, cv::Size(srcImage.cols, srcImage.rows), 0, 0, cv::INTER_LINEAR);

    cv::Mat dstimg = srcImage.clone();

    #pragma omp parallel for
    for (int h = 0; h < srcImage.rows; h++)
    {
        for (int w = 0; w < srcImage.cols; w++)
        {

            cv::Vec3b src_color = srcImage.at<cv::Vec3b>(h, w);
            int label = segmentation_map.ptr<int>(h)[w * 2];
            cv::Vec3b color = colors[label];
            dstimg.at<cv::Vec3b>(h, w) = cv::Vec3b(
                    uchar(src_color[0] * 0.2 + color[2] * 0.8),
                    uchar(src_color[1] * 0.2 + color[1] * 0.8),
                    uchar(src_color[2] * 0.2 + color[0] * 0.8)
            );
        }
    }
    return dstimg;
}

int inference_video(PPMobileSeg &model) {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "open camera failed" << std::endl;
        return -1;
    }
    while (true) {
        cv::Mat frame;
        cap >> frame;
        cv::Mat outputImg = model.inference(frame);
        cv::imshow("Output", outputImg);

        // Check for key press
        char key = (char) cv::waitKey(1);
        if (key == 27) // Press ESC key to exit
        {
            break;
        }
    }
    cap.release();
    destroyAllWindows();
    return 0;
}

void inference_image(PPMobileSeg &model, const std::string& image_path) {
    cv::Mat srcImage = cv::imread(image_path);
    cv::Mat outputImg = model.inference(srcImage);
    cv::imshow("Output", outputImg);
    cv::waitKey(0);
}

int main(int argc, char** args)
{
    if (argc < 2) {
        std::cout << "Usage: pp_mobileseg mode [image_path]" << std::endl;
        return -1;
    }
    PPMobileSeg model;
    std::string mode = args[1];

    if (mode == "video") {
        return inference_video(model);
    } else if (mode == "image") {
        std::string image_path = args[2];
        inference_image(model, image_path);
    } else {
        std::cout << "Inference mode should be in 'video' or 'image'" << std::endl;
        return -1;
    }
    return 0;
}
