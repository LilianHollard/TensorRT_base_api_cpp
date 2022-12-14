#include <iostream>
#include "engine.h"
#include <chrono> 

typedef std::chrono::high_resolution_clock Clock;

int main(){
  Options options;
  options.optBatchSizes = {2, 4, 8};
  
  Engine engine(options);
  
  const std::string onnxModelpath = "../yolov5/weights/best.onnx";
  /*bool succ = engine.build(onnxModelpath);
  
  if(!succ){
    throw std::runtime_error("Unable to build TRT engine.");
  }*/
  
  bool succ = engine.loadNetwork();
  if(!succ) {
    throw std::runtime_error("Unable to load TRT engine.");
  }
  
  //change batchSize to 1 since yolov5 require only one input 
  const size_t batchSize = 1;
  std::vector<cv::Mat> images;
  
  const std::string inputImage = "./zidane.jpg";
  auto img = cv::imread(inputImage);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  for (size_t i = 0; i < batchSize; ++i){
    images.push_back(img);
  }
  
  std::vector<std::vector<float>> featureVectors;
  succ = engine.runInference(images, featureVectors);
  if(!succ){
    throw std::runtime_error("Unable to run inference.");
  }
  
  size_t numIterations = 100;
  
  auto t1 = Clock::now();
  for (size_t i = 0; i < numIterations; ++i){
    featureVectors.clear();
    engine.runInference(images, featureVectors);
  }
  
  auto t2 = Clock::now();
  double totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  
  std::cout << "Success! Average time per inference: " << totalTime / numIterations / static_cast<float>(images.size()) <<
    " ms, for batch size of: " << images.size() << std::endl;

	return 0;
}
