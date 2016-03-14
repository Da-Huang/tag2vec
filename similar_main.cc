#include <eigen3/Eigen/Core>
#include <fstream>
#include <string>
#include <vector>

#include "model/tag2vec.h"
#include "document/score_item.h"
#include "util/logging.h"
#include "util/util.h"

namespace deeplearning {
namespace embedding {
namespace {

void PrintScoreItemVec(const std::vector<ScoreItem>& score_item_vec) {
  for (const ScoreItem& score_item : score_item_vec) {
    LOG(INFO) << score_item.ToString();
  }
}

void Test(const std::string& model_path) {
  Tag2Vec tag2vec;
  std::ifstream fin(model_path);
  Tag2Vec::Read(&fin, &tag2vec);

  std::string tag;
  std::vector<ScoreItem> ans;

  tag = "music";
  LOG(INFO) << tag << ":";
  ans = tag2vec.MostSimilar(tag2vec.TagVec(tag), 10);
  PrintScoreItemVec(ans);

  tag = "food";
  LOG(INFO) << tag << ":";
  ans = tag2vec.MostSimilar(tag2vec.TagVec(tag), 10);
  PrintScoreItemVec(ans);

  tag = "writing";
  LOG(INFO) << tag << ":";
  ans = tag2vec.MostSimilar(tag2vec.TagVec(tag), 10);
  PrintScoreItemVec(ans);

  tag = "cooking";
  LOG(INFO) << tag << ":";
  ans = tag2vec.MostSimilar(tag2vec.TagVec(tag), 10);
  PrintScoreItemVec(ans);

  tag = "organic";
  LOG(INFO) << tag << ":";
  ans = tag2vec.MostSimilar(tag2vec.TagVec(tag), 10);
  PrintScoreItemVec(ans);

  std::string words;
  words = "flashmob experiment arkansas keeping entertained assembly";
  LOG(INFO) << words << ":";
  ans = tag2vec.MostSimilar(tag2vec.Infer(util::Split(words, " "), 10), 10);
  PrintScoreItemVec(ans);

  words = "rib shack totally open years construction finally explore ri";
  LOG(INFO) << words << ":";
  ans = tag2vec.MostSimilar(tag2vec.Infer(util::Split(words, " "), 10), 10);
  PrintScoreItemVec(ans);

  words = "music";
  LOG(INFO) << words << ":";
  ans = tag2vec.MostSimilar(tag2vec.Infer(util::Split(words, " "), 10), 10);
  PrintScoreItemVec(ans);
}

}  // namespace
}  // namespace embedding
}  // namespace deeplearning

int main(int argc, char** argv) {
  deeplearning::embedding::Test("/home/dhuang/tag2vec.model");
  return 0;
}
