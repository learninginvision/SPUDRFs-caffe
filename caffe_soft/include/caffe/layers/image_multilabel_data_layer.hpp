#ifndef CAFFE_IMAGE_MULTILABEL_DATA_LAYER_HPP_
#define CAFFE_IMAGE_MULTILABEL_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ImageMultilabelDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageMultilabelDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageMultilabelDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  vector<std::pair<std::string, int *> > lines_; // string 对应 train.txt 中的图片名，int * 实现多label，其中各label间默认以空格隔离.
  int lines_id_;
};


}  // namespace caffe

#endif  // CAFFE_IMAGE_MULTILABEL_DATA_LAYER_HPP_
