#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "NvInfer.h"
#include "resizeNearestPlugin.h"
#include "maskRCNNKernels.h"

#define USE_FP16 // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0 // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1

    // stuff we know about the network and the input/output blobs
    static const int INPUT_H = Yolo::INPUT_H;
    static const int INPUT_W = Yolo::INPUT_W;
    static const int CLASS_NUM = Yolo::CLASS_NUM;
    static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1; // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
    const char *INPUT_BLOB_NAME = "data";
    const char *OUTPUT_BLOB_NAME = "prob";
    static Logger gLogger;

    static int get_width(int x, float gw, int divisor = 8)
    {
        //return math.ceil(x / divisor) * divisor

        return int(ceil((x * gw) / divisor)) * divisor;
    }

    static int get_depth(int x, float gd)
    {
        if (x == 1)
        {
            return 1;
        }
        else
        {
            return round(x * gd) > 1 ? round(x * gd) : 1;
        }
    }

    ICudaEngine *build_engine(unsigned int maxBatchSize, IBuilder *builder, DataType dt, float &gd, float &gw, std::string &wts_name)
    {
        INetworkDefinition *network = builder->createNetwork();
        // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
        ITensor *data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
        assert(data);

        std::map<std::string, Weights> weightMap = loadWeights(wts_name);

        /* ------ yolov5 backbone------ */
        auto focus0 = focus(network, weightMap, *data, 3, get_width(64, gw), 3, "model.0");
        auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
        auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
        auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
        auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(9, gd), true, 1, 0.5, "model.4");
        auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
        auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
        auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
        auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, 9, 13, "model.8");

        /* ------ yolov5 head ------ */
        auto bottleneck_csp9 = C3(network, weightMap, *spp8->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.9");
        auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");

        // auto upsample11 = network->addResize(*conv10->getOutput(0));
        // assert(upsample11);
        // upsample11->setResizeMode(ResizeMode::kNEAREST);
        // upsample11->setOutputDimensions(bottleneck_csp6->getOutput(0)->getDimensions());

        //Creating the resize plugin
        std::vector<nvinfer1::PluginField> mPluginAttributes;
        nvinfer1::PluginFieldCollection mPFC;
        float scaleFactor1 = float(bottleneck_csp6->getOutput(0)->getDimensions().d[1]/ conv10->getOutput(0)->getDimensions().d[1]);

        mPluginAttributes.clear();
        mPluginAttributes.emplace_back(nvinfer1::PluginField("scale", &scaleFactor1,
        nvinfer1::PluginFieldType::kFLOAT32, 1)); 
        mPFC.nbFields = mPluginAttributes.size();
        mPFC.fields = mPluginAttributes.data();

        auto creator = getPluginRegistry()->getPluginCreator("ResizeNearest_TRT", "1");
        auto resizeLayer = creator->createPlugin("resizeLayer0", &mPFC);
        nvinfer1::ITensor *resizelayer1_input[] = {conv10->getOutput(0)};
        //auto d2bPlugin1 = network->addPluginV2(delta_to_bbox_layer1_input, 2, *delta_to_bbox_layer1);
        auto upsample11 = network->addPluginV2(resizelayer1_input, 1, *resizeLayer);
        //auto upsample11 = resizeLayer->enqueue(1, *conv10->getOutput(0), *bottleneck_csp4->getOutput(0), nullptr, )

        ITensor *inputTensors12[] = {upsample11->getOutput(0), bottleneck_csp6->getOutput(0)};
        auto cat12 = network->addConcatenation(inputTensors12, 2);
        auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");
        auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");

        // auto upsample15 = network->addResize(*conv14->getOutput(0));
        // assert(upsample15);
        // upsample15->setResizeMode(ResizeMode::kNEAREST);
        // upsample15->setOutputDimensions(bottleneck_csp4->getOutput(0)->getDimensions());

        //RESIZE PLUGIN 2nd LAYER
        float scaleFactor2 = float(bottleneck_csp4->getOutput(0)->getDimensions().d[1]/ conv14->getOutput(0)->getDimensions().d[1]);
	    mPluginAttributes.clear();
        mPluginAttributes.emplace_back(nvinfer1::PluginField("scale", &scaleFactor2,
        nvinfer1::PluginFieldType::kFLOAT32, 1));
        mPFC.nbFields = mPluginAttributes.size();
        mPFC.fields = mPluginAttributes.data();
        auto resizeLayer2 = creator->createPlugin("resizeLayer2", &mPFC);
	    nvinfer1::ITensor *resize_layer2_input[] = {conv14->getOutput(0)};
        auto upsample15 = network->addPluginV2(resize_layer2_input, 1, *resizeLayer2);

        ITensor *inputTensors16[] = {upsample15->getOutput(0), bottleneck_csp4->getOutput(0)};
        auto cat16 = network->addConcatenation(inputTensors16, 2);

        auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

        // /* ------ detect ------ */
        IConvolutionLayer *det0 = network->addConvolution(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
        auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
        ITensor *inputTensors19[] = {conv18->getOutput(0), conv14->getOutput(0)};
        auto cat19 = network->addConcatenation(inputTensors19, 2);
        auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");
        IConvolutionLayer *det1 = network->addConvolution(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
        auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512, gw), 3, 2, 1, "model.21");
        ITensor *inputTensors22[] = {conv21->getOutput(0), conv10->getOutput(0)};
        auto cat22 = network->addConcatenation(inputTensors22, 2);
        auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
        IConvolutionLayer *det2 = network->addConvolution(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

        auto yolo = addYoLoLayer(network, weightMap, "model.24", std::vector<IConvolutionLayer *>{det0, det1, det2});
        yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
        network->markOutput(*yolo->getOutput(0));

        // Build engine
        builder->setMaxBatchSize(maxBatchSize);
        builder->setMaxWorkspaceSize(16 * (1 << 20)); // 16MB
        if (builder->platformHasFastFp16())
            builder->setFp16Mode(true);

        std::cout << "Building engine, please wait for a while..." << std::endl;
        ICudaEngine *engine = builder->buildCudaEngine(*network);
        std::cout << "Build engine successfully!" << std::endl;

        // Don't need the network any more
        network->destroy();

        // Release host memory
        for (auto &mem : weightMap)
        {
            free((void *)(mem.second.values));
        }

        return engine;
    }


    void APIToModel(unsigned int maxBatchSize, IHostMemory **modelStream, bool &is_p6, float &gd, float &gw, std::string &wts_name)
    {
        // Create builder
        IBuilder *builder = createInferBuilder(gLogger);

        // Create model to populate the network, then set the outputs and create an engine
        ICudaEngine *engine = nullptr;

        engine = build_engine(maxBatchSize, builder, DataType::kFLOAT, gd, gw, wts_name);

        assert(engine != nullptr);

        // Serialize the engine
        (*modelStream) = engine->serialize();

        // Close everything down
        engine->destroy();
        builder->destroy();
    }

    void doInference(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *input, float *output, int batchSize)
    {
        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.enqueue(batchSize, buffers, stream, nullptr);
        CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    }

    bool parse_args(int argc, char **argv, std::string &wts, std::string &engine, bool &is_p6, float &gd, float &gw, std::string &img_dir)
    {
        if (argc < 4)
            return false;
        if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7))
        {
            wts = std::string(argv[2]);
            engine = std::string(argv[3]);
            auto net = std::string(argv[4]);
            if (net[0] == 's')
            {
                gd = 0.33;
                gw = 0.50;
            }
            else if (net[0] == 'm')
            {
                gd = 0.67;
                gw = 0.75;
            }
            else if (net[0] == 'l')
            {
                gd = 1.0;
                gw = 1.0;
            }
            else if (net[0] == 'x')
            {
                gd = 1.33;
                gw = 1.25;
            }
            else if (net[0] == 'c' && argc == 7)
            {
                gd = atof(argv[5]);
                gw = atof(argv[6]);
            }
            else
            {
                return false;
            }
            if (net.size() == 2 && net[1] == '6')
            {
                is_p6 = true;
            }
        }
        else if (std::string(argv[1]) == "-d" && argc == 4)
        {
            engine = std::string(argv[2]);
            img_dir = std::string(argv[3]);
        }
        else
        {
            return false;
        }
        return true;
    }

    int main(int argc, char **argv)
    {
        cudaSetDevice(DEVICE);

        std::string wts_name = "";
        std::string engine_name = "";
        bool is_p6 = false;
        float gd = 0.0f, gw = 0.0f;
        std::string img_dir;
        if (!parse_args(argc, argv, wts_name, engine_name, is_p6, gd, gw, img_dir))
        {
            std::cerr << "arguments not right!" << std::endl;
            std::cerr << "./yolov5 -s [.wts] [.engine] [s/m/l/x/s6/m6/l6/x6 or c/c6 gd gw]  // serialize model to plan file" << std::endl;
            std::cerr << "./yolov5 -d [.engine] ../samples  // deserialize plan file and run inference" << std::endl;
            return -1;
        }

        // create a model using the API directly and serialize it to a stream
        if (!wts_name.empty())
        {
            IHostMemory *modelStream{nullptr};
            APIToModel(BATCH_SIZE, &modelStream, is_p6, gd, gw, wts_name);
            assert(modelStream != nullptr);
            std::ofstream p(engine_name, std::ios::binary);
            if (!p)
            {
                std::cerr << "could not open plan output file" << std::endl;
                return -1;
            }
            p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
            modelStream->destroy();
            return 0;
        }

        // deserialize the .engine and run inference
        std::ifstream file(engine_name, std::ios::binary);
        if (!file.good())
        {
            std::cerr << "read " << engine_name << " error!" << std::endl;
            return -1;
        }
        char *trtModelStream = nullptr;
        size_t size = 0;
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();

        std::vector<std::string> file_names;
        if (read_files_in_dir(img_dir.c_str(), file_names) < 0)
        {
            std::cerr << "read_files_in_dir failed." << std::endl;
            return -1;
        }

        // prepare input data ---------------------------
        static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
        //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        //    data[i] = 1.0;
        static float prob[BATCH_SIZE * OUTPUT_SIZE];
        IRuntime *runtime = createInferRuntime(gLogger);
        assert(runtime != nullptr);
        ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
        assert(engine != nullptr);
        IExecutionContext *context = engine->createExecutionContext();
        assert(context != nullptr);
        delete[] trtModelStream;
        assert(engine->getNbBindings() == 2);
        void *buffers[2];
        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
        const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
        assert(inputIndex == 0);
        assert(outputIndex == 1);
        // Create GPU buffers on device
        CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
        // Create stream
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        int fcount = 0;
        for (int f = 0; f < (int)file_names.size(); f++)
        {
            fcount++;
            if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size())
                continue;
            for (int b = 0; b < fcount; b++)
            {
                cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
                if (img.empty())
                    continue;
                cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H); // letterbox BGR to RGB
                int i = 0;
                for (int row = 0; row < INPUT_H; ++row)
                {
                    uchar *uc_pixel = pr_img.data + row * pr_img.step;
                    for (int col = 0; col < INPUT_W; ++col)
                    {
                        data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
                        data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                        data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                        uc_pixel += 3;
                        ++i;
                    }
                }
            }

            // Run inference
            auto start = std::chrono::system_clock::now();
            doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
            auto end = std::chrono::system_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
            std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
            for (int b = 0; b < fcount; b++)
            {
                auto &res = batch_res[b];
                nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
            }
            for (int b = 0; b < fcount; b++)
            {
                auto &res = batch_res[b];
                //std::cout << res.size() << std::endl;
                cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
                for (size_t j = 0; j < res.size(); j++)
                {
                    cv::Rect r = get_rect(img, res[j].bbox);
                    cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                    cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
                }
                cv::imwrite("_" + file_names[f - fcount + 1 + b], img);
            }
            fcount = 0;
        }

        // Release stream and buffers
        cudaStreamDestroy(stream);
        CUDA_CHECK(cudaFree(buffers[inputIndex]));
        CUDA_CHECK(cudaFree(buffers[outputIndex]));
        // Destroy the engine
        context->destroy();
        engine->destroy();
        runtime->destroy();

        // Print histogram of the output distribution
        //std::cout << "\nOutput:\n\n";
        //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
        //{
        //    std::cout << prob[i] << ", ";
        //    if (i % 10 == 0) std::cout << std::endl;
        //}
        //std::cout << std::endl;

        return 0;
    }
