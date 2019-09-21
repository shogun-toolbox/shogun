#include <shogun/io/CSVFile.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/UniformIntDistribution.h>
#include <shogun/mathematics/UniformRealDistribution.h>

#include <cstdio>
#include <cstring>
#include <random>

#include <gtest/gtest.h>

using namespace shogun;

TEST(CSVFileTest, vector_int32)
{
	int32_t seed = 100;
	int32_t len=512*512;
	SGVector<int32_t> data(len);

	std::mt19937_64 prng(seed);
  	UniformIntDistribution<int32_t> uniform_int_dist(0, len);

	for (int32_t i=0; i<len; i++)
		data[i]=uniform_int_dist(prng);

	std::shared_ptr<CSVFile> fin, fout;

	fout=std::make_shared<CSVFile>("CSVFileTest_vector_int32_output.txt",'w');
	fout->set_delimiter(' ');
	fout->set_vector(data.vector, len);
	// flush the output
	fout.reset();

	SGVector<int32_t> data_from_file;
	fin=std::make_shared<CSVFile>("CSVFileTest_vector_int32_output.txt",'r');
	fin->set_delimiter(' ');
	fin->get_vector(data_from_file.vector, data_from_file.vlen);
	EXPECT_EQ(data_from_file.vlen, len);

	for (int32_t i=0; i<data_from_file.vlen; i++)
	{
		EXPECT_EQ(data_from_file[i], data[i]);
	}
	unlink("CSVFileTest_vector_int32_output.txt");
}

TEST(CSVFileTest, vector_float64)
{
	int32_t seed = 100;
	int32_t len=128*128;
	SGVector<float64_t> data(len);

	std::mt19937_64 prng(seed);
  	UniformRealDistribution<float64_t> uniform_real_dist(0., 1.);

	for (int32_t i=0; i<len; i++)
		data[i]=uniform_real_dist(prng);

	std::shared_ptr<CSVFile> fin, fout;

	fout=std::make_shared<CSVFile>("CSVFileTest_vector_float64_output.txt",'w');
	fout->set_delimiter(' ');
	fout->set_vector(data.vector, len);
	// flush the output
	fout.reset();

	SGVector<float64_t> data_from_file;
	fin=std::make_shared<CSVFile>("CSVFileTest_vector_float64_output.txt",'r');
	fin->set_delimiter(' ');
	fin->get_vector(data_from_file.vector, data_from_file.vlen);
	EXPECT_EQ(data_from_file.vlen, len);

	for (int32_t i=0; i<data_from_file.vlen; i++)
	{
		EXPECT_NEAR(data_from_file[i], data[i], 1E-14);
	}
	unlink("CSVFileTest_vector_float64_output.txt");
}

TEST(CSVFileTest, matrix_int32)
{
	int32_t seed = 100;
	int32_t num_rows=512;
	int32_t num_cols=512;
	SGMatrix<int32_t> data(num_rows, num_cols);

	std::mt19937_64 prng(seed);
  	UniformIntDistribution<int32_t> uniform_int_dist(0, num_rows);

	for (int32_t i=0; i<num_rows; i++)
	{
		for (int32_t j=0; j<num_cols; j++)
			data(i, j)=uniform_int_dist(prng);
	}

	std::shared_ptr<CSVFile> fin, fout;

	fout=std::make_shared<CSVFile>("CSVFileTest_matrix_int32_output.txt",'w');
	fout->set_delimiter('|');
	fout->set_matrix(data.matrix, num_cols, num_rows);
	// flush the output
	fout.reset();

	SGMatrix<float64_t> data_from_file(true);
	fin=std::make_shared<CSVFile>("CSVFileTest_matrix_int32_output.txt",'r');
	fin->set_delimiter('|');
	fin->get_matrix(data_from_file.matrix, data_from_file.num_cols, data_from_file.num_rows);
	EXPECT_EQ(data_from_file.num_rows, num_rows);
	EXPECT_EQ(data_from_file.num_cols, num_cols);

	for (int32_t i=0; i<num_rows; i++)
	{
		for (int32_t j=0; j<num_cols; j++)
			EXPECT_EQ(data_from_file(i, j), data(i, j));
	}

	unlink("CSVFileTest_matrix_int32_output.txt");
}

TEST(CSVFileTest, matrix_float64)
{
	int32_t seed = 100;
	int32_t num_rows=128;
	int32_t num_cols=128;
	SGMatrix<float64_t> data(num_rows, num_cols);

	std::mt19937_64 prng(seed);
  	UniformRealDistribution<float64_t> uniform_real_dist(0., 1.);

	for (int32_t i=0; i<num_rows; i++)
	{
		for (int32_t j=0; j<num_cols; j++)
			data(i, j)=uniform_real_dist(prng);
	}

	std::shared_ptr<CSVFile> fin, fout;

	fout=std::make_shared<CSVFile>("CSVFileTest_matrix_float64_output.txt",'w');
	fout->set_delimiter('|');
	fout->set_matrix(data.matrix, num_cols, num_rows);
	// flush the output
	fout.reset();

	SGMatrix<float64_t> data_from_file(true);
	fin=std::make_shared<CSVFile>("CSVFileTest_matrix_float64_output.txt",'r');
	fin->set_delimiter('|');
	fin->get_matrix(data_from_file.matrix, data_from_file.num_cols, data_from_file.num_rows);
	EXPECT_EQ(data_from_file.num_rows, num_rows);
	EXPECT_EQ(data_from_file.num_cols, num_cols);

	for (int32_t i=0; i<num_rows; i++)
	{
		for (int32_t j=0; j<num_cols; j++)
			EXPECT_NEAR(data_from_file(i, j), data(i, j), 1E-14);
	}

	unlink("CSVFileTest_matrix_float64_output.txt");
}

TEST(CSVFileTest, string_list_char)
{
	int32_t num_lines=5;
	const char* text[] = {"It had to be U...", "U D transpose V", "I looked all around", "And finally found", "The SVD!"};

	int32_t num_str=0;
	int32_t max_line_len=0;
	SGVector<char>* lines_to_read;

	SGVector<char>* lines_to_write=SG_MALLOC(SGVector<char>, num_lines);
	for (int32_t i=0; i<num_lines; i++)
		lines_to_write[i] = SGVector<char>((char*)text[i], strlen(text[i]), false);

	std::shared_ptr<CSVFile> fin, fout;

	fout=std::make_shared<CSVFile>("CSVFileTest_string_list_char_output.txt",'w');
	fout->set_string_list(lines_to_write, num_lines);
	// flush the output
	fout.reset();

	fin=std::make_shared<CSVFile>("CSVFileTest_string_list_char_output.txt",'r');
	fin->get_string_list(lines_to_read, num_str, max_line_len);
	EXPECT_EQ(num_str, num_lines);

	for (int32_t i=0; i<num_str; i++)
	{
		for (int32_t j=0; j<lines_to_read[i].vlen; j++)
		{
			EXPECT_EQ(lines_to_read[i].vector[j], lines_to_write[i].vector[j]);
		}
	}

	SG_FREE(lines_to_write);
	SG_FREE(lines_to_read);
	unlink("CSVFileTest_string_list_char_output.txt");
}
