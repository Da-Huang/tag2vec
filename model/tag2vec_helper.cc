#include "model/tag2vec_helper.h"

#include <cmath>
#include <eigen3/Eigen/Core>
#include <omp.h>
#include <random>
#include <string>
#include <vector>

#include "document/document.h"
#include "document/tag.h"
#include "document/vocabulary.h"
#include "document/word.h"
#include "model/tag2vec.h"

namespace deeplearning {
namespace embedding {

Tag2Vec::Random::Random() {
  std::random_device rd;
  engine_.seed(rd());
  sample_ = new std::uniform_real_distribution<float>(0, 1);
}

Tag2Vec::Random::Random(size_t window) : Random() {
  window_ = new std::uniform_int_distribution<size_t>(0, window);
}

Tag2Vec::Random::~Random() {
  if (sample_) delete sample_;
  if (window_) delete window_;
}

float Tag2Vec::Random::Sample() {
  float ans = 0;
  #pragma omp critical(Tag2VecRandom)
  ans = (*sample_)(engine_);
  return ans;
}

float Tag2Vec::Random::Window() {
  float ans = 0;
  #pragma omp critical(Tag2VecRandom)
  ans = (*window_)(engine_);
  return ans;
}

void BuildWordVocabulary(DocumentIterator* iterator, size_t min_count,
                         float sample, Vocabulary* vocabulary) {
  const Document* document;
  while ((document = iterator->NextDocument())) {
    for (const std::string& word : document->words()) {
      vocabulary->AddItem<Word>(word);
    }
    delete document;
  }
  vocabulary->Build(min_count);

  if (sample) {
    float threshold = sample * vocabulary->num_retained();
    for (size_t i = 0; i < vocabulary->items_size(); ++i) {
      Word* word = (Word*)vocabulary->item(i);
      float probability =
          (std::sqrt(word->count() / threshold) + 1) * (threshold / word->count());
      word->set_probability(std::min(probability, 1.0f));
    }
  }
}

void BuildTagVocabulary(DocumentIterator* iterator, Vocabulary* vocabulary) {
  const Document* document;
  while ((document = iterator->NextDocument())) {
    for (const std::string& tag : document->tags()) {
      vocabulary->AddItem<Tag>(tag);
    }
    delete document;
  }
  vocabulary->Build();
}

void GetVocabularyItemVec(const Vocabulary& vocabulary,
                          const std::vector<std::string>& item_strs,
                          std::vector<const Vocabulary::Item*>* item_vec) {
  for (const std::string& item_str : item_strs) {
    const Vocabulary::Item* item = vocabulary.item(item_str);
    if (item) {
      item_vec->push_back(item);
    }
  }
}

void TrainSgPair(Tag2Vec::RMatrixXf::RowXpr input, Tag2Vec::RMatrixXf& output,
                 const std::vector<bool>& codes,
                 const std::vector<size_t>& points, float alpha,
                 bool update_output) {
  Eigen::VectorXf neu1e(input.size());
  neu1e.setZero();
  for (size_t i = 0; i < codes.size(); ++i) {
    const float f = 1.0f / (1 + std::exp(-input.dot(output.row(points[i]))));
    const float g = (codes[i] - f) * alpha;
    // Applies old output value to update.
    neu1e += g * output.row(points[i]);
    if (update_output) {
      output.row(points[i]) += g * input;
    }
  }
  input += neu1e;
}

}  // namespace embedding
}  // namespace deeplearning
