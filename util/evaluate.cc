#include "util/evaluate.h"

#include <string>
#include <unordered_set>
#include <vector>

#include "document/score_item.h"
#include "util/logging.h"

namespace deeplearning {
namespace embedding {
namespace util {

void Evaluate(Model* model, DocumentIterator* test_data) {
  std::vector<Document> documents;
  const Document* document;
  while ((document = test_data->NextDocument())) {
    documents.push_back(*document);
    delete document;
  }
  Evaluate(model, documents);
}

void Evaluate(Model* model, const std::vector<Document>& test_data) {
  size_t num_correct = 0, num_suggested = 0;

  #pragma omp parallel for
  for (size_t i = 0; i < test_data.size(); ++i) {
    std::unordered_set<std::string> tag_set(test_data[i].tags().begin(),
                                            test_data[i].tags().end());
    std::vector<ScoreItem> suggested_tags =
        model->Suggest(test_data[i].words());
    CHECK_EQ(1, suggested_tags.size()) << "Should only suggest 1 item.";
    if (tag_set.find(suggested_tags[0].item()) != tag_set.end()) {
      #pragma omp atomic update
      ++num_correct;
    }

    #pragma omp atomic update
    num_suggested += suggested_tags.size();

    static const size_t DISPLAY_NUM = 1000;
    if (num_suggested / DISPLAY_NUM >
        (num_suggested - test_data[i].tags().size()) / DISPLAY_NUM) {
      LOG(INFO) << "Suggested " << num_suggested
                << " tags. p@1=" << (float)num_correct / num_suggested;
    }
  }
  LOG(INFO) << "Suggested " << num_suggested
            << " tags. p@1=" << (float)num_correct / num_suggested;
}


}  // namespace util
}  // namespace embedding
}  // namespace deeplearning
