#include "model/tw2vec.h"

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <random>
#include <string>
#include <vector>

#include "document/document.h"
#include "document/memory_document_iterator.h"
#include "document/score_item.h"
#include "document/tag.h"
#include "document/vocabulary.h"
#include "document/word.h"
#include "model/tag2vec.h"
#include "model/tag2vec_helper.h"
#include "util/huffman.h"
#include "util/io.h"
#include "util/logging.h"

namespace deeplearning {
namespace embedding {

Tw2Vec::Tw2Vec() {
  Initialize();
}

Tw2Vec::Tw2Vec(size_t window, size_t layer_size, size_t min_count,
               float init_alpha, float min_alpha)
    : window_(window),
      layer_size_(layer_size),
      min_count_(min_count),
      init_alpha_(init_alpha),
      min_alpha_(min_alpha) {
  Initialize();
}

Tw2Vec::~Tw2Vec() {
  delete random_;
}

void Tw2Vec::Train(const std::vector<Document>& documents, size_t iter) {
  CHECK(!has_trained_) << "Tw2Vec has already been trained.";
  MemoryDocumentIterator iterator(documents);
  BuildTagVocabulary(&iterator, &tag_vocab_);
  iterator.Reset();
  BuildWordVocabulary(&iterator, min_count_, 0 /* sample */, &word_vocab_);
  word_huffman_.Build(word_vocab_.items());

  CHECK(tag_vocab_.items_size() >= 1)
      << "Size of tag_vocab should be at least 1.";
  CHECK(word_vocab_.items_size() >= 1)
      << "Size of word_vocab should be at least 1.";
  LOG(INFO) << "Vocabularies for words and tags have been built.";

  // Initializes weights.
  std::uniform_real_distribution<float> dist(-0.5, 0.5);
  const auto uniform =
      [&dist, this](size_t) { return dist(this->random_->engine()); };
  size_t tag_dim = tag_vocab_.items_size();
  tagi_ = Tag2Vec::RMatrixXf::NullaryExpr(tag_dim, layer_size_, uniform) /
          layer_size_;
  size_t word_dim = word_vocab_.items_size() - 1;
  wordo_ = Tag2Vec::RMatrixXf::NullaryExpr(word_dim, layer_size_, uniform) /
           layer_size_;
  wordi_ = Tag2Vec::RMatrixXf::NullaryExpr(word_vocab_.items_size(),
                                           layer_size_, uniform) /
           layer_size_;
  LOG(INFO) << "Weights have been initialized.";

  float alpha = init_alpha_;
  size_t num_words = 0;
  for (size_t t = 0; t < iter; ++t) {
    #pragma omp parallel for
    for (size_t doc = 0; doc < documents.size(); ++doc) {
      const Document& document = documents[doc];

      // Gets words and tags.
      std::vector<const Vocabulary::Item*> word_vec, tag_vec;
      GetVocabularyItemVec(word_vocab_, document.words(), &word_vec);
      GetVocabularyItemVec(tag_vocab_, document.tags(), &tag_vec);

      // Trains document.
      for (const Vocabulary::Item* tag : tag_vec) {
        for (size_t i = 0; i < word_vec.size(); ++i) {

          size_t reduced_window = window_ - random_->Window();
          size_t left = i >= reduced_window ? i - reduced_window : 0;
          size_t right = std::min(i + reduced_window + 1, word_vec.size());
          for (size_t j = left; j < right; ++j) {
            if (i == j) continue;

            TrainSgPair(tagi_.row(tag->index()), wordo_,
                        word_huffman_.codes(word_vec[i]->index()),
                        word_huffman_.points(word_vec[i]->index()), alpha,
                        true);

            TrainSgPair(wordi_.row(word_vec[i]->index()), wordo_,
                        word_huffman_.codes(word_vec[j]->index()),
                        word_huffman_.points(word_vec[j]->index()), alpha,
                        true);
          }
        }
      }

      #pragma omp atomic update
      num_words += document.words().size();
      alpha = init_alpha_ -
              (init_alpha_ - min_alpha_) * num_words /
                  word_vocab_.num_original() / iter;

      static const size_t DISPLAY_NUM = 100000;
      if (num_words / DISPLAY_NUM >
          (num_words - document.words().size()) / DISPLAY_NUM) {
        LOG(INFO) << "Iter" << t << ": Processed " << num_words << "/"
                  << word_vocab_.num_original() * iter
                  << " words. [alpha=" << alpha << "].";
      }
    }
    LOG(INFO) << "Iteration " << t << " has finished.";
  }
  has_trained_ = true;
}

void Tw2Vec::Train(DocumentIterator* iterator, size_t iter) {
  CHECK(!has_trained_) << "Tw2Vec has already been trained.";
  std::vector<Document> documents;
  const Document* document;
  while ((document = iterator->NextDocument())) {
    documents.push_back(*document);
    delete document;
  }
  Train(documents, iter);
  has_trained_ = true;
}

std::vector<ScoreItem> Tw2Vec::Suggest(const std::vector<std::string>& words) {
  return MostSimilar(Infer(words, 5), 1);
}

bool Tw2Vec::ContainsTag(const std::string& tag) const {
  return tag_vocab_.item(tag);
}

Eigen::VectorXf Tw2Vec::TagVec(const std::string& tag) const {
  return tagi_.row(tag_vocab_.item(tag)->index());
}

std::vector<ScoreItem> Tw2Vec::MostSimilar(const Eigen::VectorXf& v,
                                            size_t limit) const {
  const auto greater_score_item =
      [](const ScoreItem& si1, const ScoreItem& si2) { return si1 > si2; };
  std::vector<ScoreItem> ans;
  for (const Vocabulary::Item* item : tag_vocab_.items()) {
    Tag2Vec::RMatrixXf::ConstRowXpr tagv = tagi_.row(item->index());
    float score = v.dot(tagv) / v.norm() / tagv.norm();
    ans.emplace_back(item->text(), score);
    std::push_heap(ans.begin(), ans.end(), greater_score_item);
    if (ans.size() > limit) {
      std::pop_heap(ans.begin(), ans.end(), greater_score_item);
      ans.pop_back();
    }
  }
  std::sort(ans.begin(), ans.end(), greater_score_item);
  return ans;
}

Eigen::RowVectorXf Tw2Vec::Infer(const std::vector<std::string>& words,
                                  size_t iter) {
  std::uniform_real_distribution<float> dist(-0.5, 0.5);
  const auto uniform =
      [&dist, this](size_t) { return dist(this->random_->engine()); };
  Tag2Vec::RMatrixXf ans =
      Tag2Vec::RMatrixXf::NullaryExpr(1, layer_size_, uniform) / layer_size_;

  // Gets words.
  std::vector<const Vocabulary::Item*> word_vec;
  GetVocabularyItemVec(word_vocab_, words, &word_vec);

  float alpha = init_alpha_;

  for (size_t t = 0; t < iter; ++t) {
    for (const Vocabulary::Item* word : word_vec) {
      TrainSgPair(ans.row(0), wordo_, word_huffman_.codes(word->index()),
                  word_huffman_.points(word->index()), alpha, false);
    }
    alpha = init_alpha_ - (init_alpha_ - min_alpha_) * (t + 1) / iter;
  }
  return ans;
}

void Tw2Vec::Write(std::ostream* out) const {
  CHECK(has_trained_) << "Tw2Vec has not been trained.";
  util::WriteBasicItem(out, window_);
  util::WriteBasicItem(out, layer_size_);
  util::WriteBasicItem(out, min_count_);
  util::WriteBasicItem(out, init_alpha_);
  util::WriteBasicItem(out, min_alpha_);

  word_vocab_.Write(out);
  tag_vocab_.Write(out);
  word_huffman_.Write(out);

  util::WriteMatrix(out, tagi_);
  util::WriteMatrix(out, wordi_);
  util::WriteMatrix(out, wordo_);
}

void Tw2Vec::Read(std::istream* in, Tw2Vec* tag2vec) {
  CHECK(!tag2vec->has_trained_) << "Tw2Vec has been trained.";
  util::ReadBasicItem(in, &tag2vec->window_);
  util::ReadBasicItem(in, &tag2vec->layer_size_);
  util::ReadBasicItem(in, &tag2vec->min_count_);
  util::ReadBasicItem(in, &tag2vec->init_alpha_);
  util::ReadBasicItem(in, &tag2vec->min_alpha_);
  tag2vec->Initialize();

  Vocabulary::Read<Word>(in, &tag2vec->word_vocab_);
  Vocabulary::Read<Tag>(in, &tag2vec->tag_vocab_);
  util::Huffman::Read(in, &tag2vec->word_huffman_);

  util::ReadMatrix(in, &tag2vec->tagi_);
  util::ReadMatrix(in, &tag2vec->wordi_);
  util::ReadMatrix(in, &tag2vec->wordo_);

  tag2vec->has_trained_ = true;
}

std::string Tw2Vec::ConfigString() const {
  std::string ans;
  ans += "window=" + std::to_string(window_) + ";";
  ans += "layer_size=" + std::to_string(layer_size_) + ";";
  ans += "min_count=" + std::to_string(min_count_) + ";";
  ans += "init_alpha=" + std::to_string(init_alpha_) + ";";
  ans += "min_alpha=" + std::to_string(min_alpha_) + ";";
  return ans;
}

void Tw2Vec::Initialize() {
  CHECK(min_alpha_ < init_alpha_)
      << "init_alpha should not be less than min_alpha.";
  CHECK(0 < min_alpha_) << "min_alpha should not be greater than min_alpha.";
  CHECK(init_alpha_ <= 0.1) << "init_alpha should be no more than 0.01.";
  CHECK(0 <= window_ && window_ <= 20) << "window should be within [0, 20]";
  if (random_) delete random_;
  random_ = new Tag2Vec::Random(window_);
}

}  // namespace embedding
}  // namespace deeplearning
