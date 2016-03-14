#include <eigen3/Eigen/Core>
#include <fstream>
#include <omp.h>
#include <string>

#include "document/standard_document_iterator.h"
#include "model/tw2vec.h"
#include "util/evaluate.h"
// #include "util/logging.h"

namespace deeplearning {
namespace embedding {
namespace {

void Evaluate(const std::string& model_path,
              const std::string& test_data_path) {
  Tw2Vec tw2vec;
  std::ifstream model_in(model_path);
  Tw2Vec::Read(&model_in, &tw2vec);

  std::ifstream test_data_in(test_data_path);
  StandardDocumentIterator iterator(&test_data_in);
  util::Evaluate(&tw2vec, &iterator);
}

}  // namespace
}  // namespace embedding
}  // namespace deeplearning

int main(int argc, char** argv) {
  Eigen::initParallel();
  omp_set_num_threads(4);
  deeplearning::embedding::Evaluate("/home/dhuang/tw2vec.model",
                                    "/home/dhuang/twitter.big.test.txt");
  return 0;
}
