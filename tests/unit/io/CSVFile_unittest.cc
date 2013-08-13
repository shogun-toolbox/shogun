#include <shogun/io/CSVFile.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGString.h>
#include <shogun/lib/SGMatrix.h>

#include <cstdio>
#include <cstring>

#include <gtest/gtest.h>

using namespace shogun;

TEST(CSVFileTest, read_matrix)
{
	int32_t num_lines=5;

	int32_t num_rows=4;
	int32_t num_cols=3;
	const char* lines[]={"Header (should be skipped)", "Not header (should be skipped too)",
				"1.0000|1.0001|1.0002|1.0003",
				"1.0004|1.0005|1.0006|1.0007",
				"1.0008|1.0009|1.0010|1.0011"};

	float64_t data[]={1.0000, 1.0001, 1.0002, 1.0003,
				1.0004, 1.0005, 1.0006, 1.0007,
				1.0008, 1.0009, 1.0010, 1.0011};

	SGMatrix<float64_t> a=SGMatrix<float64_t>(data, num_rows, num_cols, false);

	FILE* fout=fopen("csvfile_test.csv","w");
	for (int32_t i=0; i<num_lines; i++)
		fprintf(fout, "%s\n", lines[i]);
	fclose(fout);

	SGMatrix<float64_t> tmp(true);
	CCSVFile* fin;

	fin=new CCSVFile("csvfile_test.csv",'r', NULL);
	fin->set_delimiter('|');
	fin->set_lines_to_skip(2);

	fin->get_matrix(tmp.matrix, tmp.num_rows, tmp.num_cols);
	EXPECT_EQ(tmp.num_rows, num_rows);
	EXPECT_EQ(tmp.num_cols, num_cols);

	for (int32_t i=0; i<tmp.num_rows; i++)
	{
		for (int32_t j=0; j<tmp.num_cols; j++)
		{
			EXPECT_EQ(tmp(i, j), a(i, j));
		}
	}
	SG_UNREF(fin);
}

TEST(CSVFileTest, write_matrix_real)
{
	int32_t num_rows=4;
	int32_t num_cols=3;

	float64_t data[]={1.0000, 1.0001, 1.0002, 1.0003,
				1.0004, 1.0005, 1.0006, 1.0007,
				1.0008, 1.0009, 1.0010, 1.0011};

	SGMatrix<float64_t> a=SGMatrix<float64_t>(data, num_rows, num_cols, false);

	CCSVFile* fin;
	CCSVFile* fout;

	SGMatrix<float64_t> tmp(true);

	fout=new CCSVFile("csvfile_test.csv",'w', NULL);
	fout->set_delimiter('|');
	fout->set_matrix(data, num_rows, num_cols);
	SG_UNREF(fout);

	fin=new CCSVFile("csvfile_test.csv",'r', NULL);
	fin->set_delimiter('|');

	fin->get_matrix(tmp.matrix, tmp.num_rows, tmp.num_cols);
	EXPECT_EQ(tmp.num_rows, num_rows);
	EXPECT_EQ(tmp.num_cols, num_cols);

	for (int32_t i=0; i<tmp.num_rows; i++)
	{
		for (int32_t j=0; j<tmp.num_cols; j++)
		{
			EXPECT_EQ(tmp(i, j), a(i, j));
		}
	}
	SG_UNREF(fin);
}

TEST(CSVFileTest, write_matrix_int)
{
	int32_t num_rows=4;
	int32_t num_cols=3;

	int32_t data[]={1, 2, 3, 4,
				5, 6, 7, 8,
				9, 10, 11, 12};

	SGMatrix<int32_t> a=SGMatrix<int32_t>(data, num_rows, num_cols, false);

	CCSVFile* fin;
	CCSVFile* fout;

	SGMatrix<int32_t> tmp(true);

	fout=new CCSVFile("csvfile_test.csv",'w', NULL);
	fout->set_delimiter('|');
	fout->set_matrix(data, num_rows, num_cols);
	SG_UNREF(fout);

	fin=new CCSVFile("csvfile_test.csv",'r', NULL);
	fin->set_delimiter('|');

	fin->get_matrix(tmp.matrix, tmp.num_rows, tmp.num_cols);
	EXPECT_EQ(tmp.num_rows, num_rows);
	EXPECT_EQ(tmp.num_cols, num_cols);

	for (int32_t i=0; i<tmp.num_rows; i++)
	{
		for (int32_t j=0; j<tmp.num_cols; j++)
		{
			EXPECT_EQ(tmp(i, j), a(i, j));
		}
	}
	SG_UNREF(fin);
}

TEST(CSVFileTest, write_vector_int)
{
	int32_t nlen=12;

	int32_t data[]={1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12};

	CCSVFile* fin;
	CCSVFile* fout;

	SGVector<int32_t> tmp(true);

	fout=new CCSVFile("csvfile_test.csv",'w', NULL);
	fout->set_delimiter(' ');
	fout->set_vector(data, nlen);
	SG_UNREF(fout);

	fin=new CCSVFile("csvfile_test.csv",'r', NULL);
	fin->set_delimiter(' ');

	fin->get_vector(tmp.vector, tmp.vlen);
	EXPECT_EQ(tmp.vlen, nlen);

	for (int32_t i=0; i<tmp.vlen; i++)
	{
		EXPECT_EQ(tmp[i], data[i]);
	}
	SG_UNREF(fin);
}

TEST(CSVFileTest, read_write_string_list)
{
	int32_t num_lines=5;
	const char* text[] = {"It had to be U...", "U D transpose V", "I looked all around", "And finally found", "The SVD!"};

	int32_t num_str=0;
	int32_t max_line_len=0;
	SGString<char>* lines_to_read;	

	SGString<char>* lines_to_write=SG_MALLOC(SGString<char>, num_lines);
	for (int32_t i=0; i<num_lines; i++)
	{
		lines_to_write[i].string=SG_MALLOC(char, strlen(text[i])+1);
		strcpy(lines_to_write[i].string, text[i]);
		lines_to_write[i].slen=strlen(text[i]);
	}

	CCSVFile* fin;
	CCSVFile* fout;	

	fout=new CCSVFile("csvfile_test.csv",'w', NULL);
	fout->set_string_list(lines_to_write, num_lines);
	SG_UNREF(fout);

	fin=new CCSVFile("csvfile_test.csv",'r', NULL);
	fin->get_string_list(lines_to_read, num_str, max_line_len);
	EXPECT_EQ(num_str, num_lines);

	for (int32_t i=0; i<num_str; i++)
	{
		for (int32_t j=0; j<lines_to_read[i].slen; j++)
			EXPECT_EQ(lines_to_read[i].string[j], lines_to_write[i].string[j]);
	}
	SG_UNREF(fin);	
}
