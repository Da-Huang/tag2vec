#ifndef UTIL_UTIL_H_
#define UTIL_UTIL_H_

#include <functional>
#include <string>
#include <vector>

namespace deeplearning {
namespace embedding {
namespace util {

// Gets the base path of file_path. '/' will be preferred to be added to back.
std::string BasePath(const std::string& file_path);

// Gets the filename from the file_path.
std::string Filename(const std::string& file_path);

// Joins path with filename.
std::string JoinPath(const std::string& path, const std::string& filename);

// Reads content from the file specified by path.
std::string ReadContent(const std::string& path);

// Splits str into a string vector.
//
// -str is the string to be split.
// -delim is the delimiters used to split str.
//
// E.g. If delim = ",;", str = ",a,;b;c,d,,e;,;;", it returns {"a", "b", "c",
// "d", "e"}.
std::vector<std::string> Split(const std::string& str, const char* delim);

// Executes system command. It crashes, when it fails.
void System(const std::string& command);

// Traverses all files under path recursively.
void TraversePath(const std::string& path,
                  const std::function<void(const std::string&)>& func);

}  // namespace util
}  // namespace embedding
}  // namespace deeplearning

#endif  // UTIL_UTIL_H_
