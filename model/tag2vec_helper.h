#ifndef MODEL_TAG2VEC_HELPER_H_
#define MODEL_TAG2VEC_HELPER_H_

#include <eigen3/Eigen/Core>
#include <random>
#include <string>
#include <vector>

#include "document/document.h"
#include "document/vocabulary.h"
#include "model/tag2vec.h"

namespace deeplearning {
namespace embedding {

class Tag2Vec::Random final {
 public:
  Random();
  Random(size_t window);
  Random(const Random& random) = delete;
  ~Random();

  std::mt19937_64& engine() { return engine_; }

  float Sample();

  float Window();

 private:
  std::mt19937_64 engine_;
  std::uniform_real_distribution<float>* sample_ = nullptr;  // OWNED
  std::uniform_int_distribution<size_t>* window_ = nullptr;  // OWNED
};

void BuildWordVocabulary(DocumentIterator* iterator, size_t min_count,
                         float sample, Vocabulary* vocabulary);

void BuildTagVocabulary(DocumentIterator* iterator, Vocabulary* vocabulary);

void GetVocabularyItemVec(const Vocabulary& vocabulary,
                          const std::vector<std::string>& item_strs,
                          std::vector<const Vocabulary::Item*>* item_vec);

void TrainSgPair(Tag2Vec::RMatrixXf::RowXpr input, Tag2Vec::RMatrixXf& output,
                 const std::vector<bool>& codes,
                 const std::vector<size_t>& points, float alpha,
                 bool update_output);

}  // namespace embedding
}  // namespace deeplearning

#endif
