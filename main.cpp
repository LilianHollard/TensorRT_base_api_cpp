#include <iostream>
#include "engine.h"
#include <chrono> 
#include <cmath>
#include <opencv2/core.hpp>

typedef std::chrono::high_resolution_clock Clock;

cv::Mat format_yolov5(const cv::Mat &source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}


int main(){
  Options options;
  options.optBatchSizes = {1};
  
  Engine engine(options);
  
  const std::string onnxModelpath = "./best.onnx";
  bool succ = engine.build(onnxModelpath);
  
  if(!succ){
    throw std::runtime_error("Unable to build TRT engine.");
  }
  
  succ = engine.loadNetwork();
  if(!succ) {
    throw std::runtime_error("Unable to load TRT engine.");
  }
  
  //change batchSize to 1 since yolov5 require only one input 
  const size_t batchSize = 1;
  std::vector<cv::Mat> images;

  const std::string inputImage = "./25_MN28_2.JPG";
  auto img1 = cv::imread(inputImage);
  auto img = format_yolov5(img1);


  
  /*def letterbox */
  int new_shape = 640;
  int stride = 32;
  float r_cols = static_cast<float>(new_shape) / static_cast<float>(img.cols);
  float r_rows = static_cast<float>(new_shape) / static_cast<float>(img.rows);
  
  float r = std::min(r_cols, r_rows);
  int new_unpad_cols = static_cast<int>(std::round(img.cols*r));
  int new_unpad_rows = static_cast<int>(std::round(img.rows*r));

  int dw = new_unpad_cols - new_unpad_rows;
  int dh = new_unpad_rows - new_unpad_cols;
  dw = dw % stride; //Returns the element-wise remainder of division
  dh = dh % stride;
  //divide padding into 2 sides
  dw = dw / 2;
  dh = dh / 2;
  cv::Mat resized;
  if(img.cols != new_unpad_cols && img.rows != new_unpad_rows){
    cv::resize(img, resized, cv::Size(640, 640), 0,0, cv::INTER_LINEAR);
  }
  //cv::flip(resized, rez
  /*int top = static_cast<int>(std::round(dh - 0.1));
  int bottom = static_cast<int>(std::round(dh + 0.1));
  
  int left = static_cast<int>(std::round(dw - 0.1));
  int right = static_cast<int>(std::round(dw + 0.1));
  cv::Mat copyMakeBordered;
  cv::copyMakeBorder(resized, copyMakeBordered, top, bottom, left, right, cv::BORDER_CONSTANT, (114, 114, 114)); */// add border
  //std::cout << copyMakeBordered.rows << std::endl;
  /**/
  
  float x_factor = static_cast<float>(img.cols) / static_cast<float>(resized.cols);
  float y_factor = static_cast<float>(img.rows) / static_cast<float>(resized.rows);
  x_factor = 1.0;
  y_factor = 1.0;
  //float x_factor = 1.0;
  //float y_factor = 1.0;
  /*std::cout << x_factor << std::endl;
  std::cout << y_factor << std::endl;
  */
  std::cout << resized.cols << " - " << resized.rows << std::endl;
  cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
  
  for (size_t i = 0; i < batchSize; ++i){
    images.push_back(resized);
  }
  
  //images.push_back(copyMakeBordered);
  std::vector<std::vector<float>> featureVectors;
  succ = engine.runInference(images, featureVectors);
  if(!succ){
    throw std::runtime_error("Unable to run inference.");
  }
  
  
  size_t numIterations = 10;
  auto t1 = Clock::now();
  for (size_t i = 0; i < numIterations; ++i){
    featureVectors.clear();
    engine.runInference(images, featureVectors);
  }
  
  auto t2 = Clock::now();
  
  /*=============================================================================================*/
  const int dimensions = 6;
  int rows = 25200; //25200 is the default size for 640.
  rows = rows / dimensions;
  std::cout << featureVectors[0].size() << std::endl;
  std::vector<cv::Rect> boxes;
  
  
  //std::vector<int> nms_result;
  //cv::dnn::NMSBoxes(boxes, 0.4, 0.4, 0.4, nms_result);
  
  
  
  //cv::resize(resized, resized, cv::Size(640, 480), 0,0, cv::INTER_LINEAR);
  for(int i = 0; i < rows; ++i){
    int index = dimensions*i + 4;//width* y + x
    float x = featureVectors.data()[0].data()[dimensions*i+0];
    float y = featureVectors.data()[0].data()[dimensions*i+1];
    float w = featureVectors.data()[0].data()[dimensions*i+2];
    float h = featureVectors.data()[0].data()[dimensions*i+3];
    float confidence = featureVectors.data()[0].data()[index];
    float cls = featureVectors.data()[0].data()[dimensions*i+5];


    //compute conf
    float conf = confidence * cls;
  
    if(confidence > 0.2){
      int x1 = x - w / 2;
      int y1 = y - h / 2;
      int x2 = x + w / 2;
      int y2 = y + h / 2;
      //int left = int((x - (0.5 * w)) * x_factor);
      //int top = int((y - (0.5 * h)) * y_factor);
      //int width = int(w * x_factor);
      //int height = int(h * y_factor);
      
      
      cv::rectangle(resized, cv::Point(x1, y1), cv::Point(x2, y2), (255,178,50), cv::LINE_4);
      //std::cout << left << " " << top << " " << width << " " <<  height << " " << confidence <<" " << cls << std::endl;
      //boxes.push_back(cv::Rect(left, top, width, height));
      //cv::rectangle(img, cv::Point(x,y), cv::Point(x+w,y+h), (255, 178, 50), cv::LINE_4);
    }
  }
  
  //auto box = cv::dnn::NMSBoxes(boxes, 0.2, 0.01);
  int countfv = 0;
  /*for(auto i: featureVectors){
    std::cout << "Feature Vectors" << std::endl;
    for(auto f: i){
      std::cout << " - " << f;
      std::cout << std::endl;
      countfv++;
    }
  }*/
  

  /*=============================================================================================*/
  
  
  //cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  cv::imwrite("25_feature.jpg", resized);

  double totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  
  std::cout << "Success! Average time per inference: " << totalTime / numIterations / static_cast<float>(images.size()) <<
    " ms, for batch size of: " << images.size() << std::endl;

	return 0;
}
