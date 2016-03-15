#ifndef MODEL_TW2VEC_H_
#define MODEL_TW2VEC_H_

#include <eigen3/Eigen/Core>
#include <iostream>
#include <string>
#include <vector>

#include "document/document.h"
#include "document/score_item.h"
#include "document/tag.h"
#include "document/vocabulary.h"
#include "document/word.h"
#include "model/model.h"
#include "model/tag2vec.h"
#include "util/huffman.h"

namespace deeplearning {
namespace embedding {

class Tw2Vec final : public Model {
 public:
  Tw2Vec();
  Tw2Vec(size_t window, size_t layer_size, size_t min_count, float init_alpha,
         float min_alpha);
  Tw2Vec(const Tw2Vec& tag2vec) = delete;
  ~Tw2Vec() override;

  void Train(const std::vector<Document>& documents, size_t iter);
  void Train(DocumentIterator* iterator, size_t iter) override;

  std::vector<ScoreItem> Suggest(
      const std::vector<std::string>& words) override;

  bool ContainsTag(const std::string& tag) const;
  Eigen::VectorXf TagVec(const std::string& tag) const;
  std::vector<ScoreItem> MostSimilar(const Eigen::VectorXf& v,
                                     size_t limit) const;

  Eigen::RowVectorXf Infer(const std::vector<std::string>& words, size_t iter);

  void Write(std::ostream* out) const override;
  static void Read(std::istream* in, Tw2Vec* tag2vec);

  std::string ConfigString() const;

 private:
  void Initialize();

 private:
  size_t window_ = 5;
  size_t layer_size_ = 500;
  size_t min_count_ = 0;
  float init_alpha_ = 0.025;
  float min_alpha_ = 0.0001;

  bool has_trained_ = false;

  Tag2Vec::Random* random_ = nullptr;  // OWNED

  Vocabulary word_vocab_, tag_vocab_;
  util::Huffman word_huffman_, tag_huffman_;

  Tag2Vec::RMatrixXf tagi_, wordi_, tago_, wordo_;
};

}  // namespace embedding
}  // namespace deeplearning

#endif  // MODEL_TW2VEC_H_
