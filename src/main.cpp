#include <iostream>
#include <Eigen/Dense>

//using namespace Eigen;

int main(int argc, char* argv[])
{
	// Eigen variable
	Eigen::MatrixXd matrix_input_a;
	Eigen::MatrixXd matrix_input_b;
	Eigen::MatrixXd matrix_output;

	matrix_input_a = Eigen::MatrixXd(2, 1);
	matrix_input_b = Eigen::MatrixXd(1, 2);
	matrix_output = Eigen::MatrixXd(2, 2);

	matrix_input_a(0, 0) = 2;
	matrix_input_a(1, 0) = 1;

	matrix_input_b(0, 0) = 2;
	matrix_input_b(0, 1) = 1;

	// multiply
	matrix_output = matrix_input_a * matrix_input_b;

	std::cout << "input ( matrix_input_a ) " << std::endl;
	std::cout << matrix_input_a << std::endl;
	std::cout << "input ( matrix_input_b ) " << std::endl;
	std::cout << matrix_input_b << std::endl;
	std::cout << "output ( matrix_output, matrix_input_a * matrix_inpupt_b ) " << std::endl;
	std::cout << matrix_output << std::endl;


	return 0;
}