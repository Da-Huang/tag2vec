#include <eigen3/Eigen/Core>
#include <iostream>

int main(int argc, char** argv) {
  using RMatrixXf =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  RMatrixXf m(3, 3);
  m << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  std::cout << "m=" << std::endl << m << std::endl;
  RMatrixXf::ColXpr cv = m.col(1);
  RMatrixXf::RowXpr v = m.row(1);
  v.setOnes();
  cv.setOnes();
  std::cout << "v=" << std::endl << v << std::endl;
  std::cout << "m=" << std::endl << m << std::endl;

  RMatrixXf z(1, 9);
  z.setOnes();
  std::cout << "z=" << std::endl << z << std::endl;
  Eigen::VectorXf p(9);
  p.setOnes();
  std::cout << "p=" << std::endl << p << std::endl;
  z.row(0) += p;
  std::cout << "z=" << std::endl << z << std::endl;

  return 0;
}
