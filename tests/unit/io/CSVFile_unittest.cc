#include <shogun/io/CSVFile.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

#include <cstdio>

#include <gtest/gtest.h>

using namespace shogun;

TEST(CSVFileTest, read_matrix)
{
	int32_t nlines=5;

	int32_t nvecs=3;
	int32_t nfeats=4;
	const char* lines[]={"Header (should be skipped)", "Not header (should be skipped too)",
				"1.0000|1.0001|1.0002|1.0003",
				"1.0004|1.0005|1.0006|1.0007",
				"1.0008|1.0009|1.0010|1.0011"};

	float64_t c_order_data[]={1.000, 1.0004, 1.0008, 
					1.0001, 1.0005, 1.0009,
					1.0002, 1.0006, 1.0010,
					1.0003, 1.0007, 1.0011};

	float64_t fortran_order_data[]={1.0000, 1.0001, 1.0002, 1.0003,
				1.0004, 1.0005, 1.0006, 1.0007,
				1.0008, 1.0009, 1.0010, 1.0011};

	SGMatrix<float64_t> a=SGMatrix<float64_t>(const_cast<float64_t* >(fortran_order_data), nfeats, nvecs, false);
	SGMatrix<float64_t> b=SGMatrix<float64_t>(const_cast<float64_t* >(c_order_data), nvecs, nfeats, false);

	FILE* fout=fopen("csvfile_test.csv","w");
	for (int32_t i=0; i<nlines; i++)
		fprintf(fout, "%s\n", lines[i]);
	fclose(fout);

	SGMatrix<float64_t> tmp(true);
	CCSVFile* fin;

	// try read in fortran order
	fin=new CCSVFile("csvfile_test.csv",'r', NULL, '|', '"');
	fin->skip_lines(2);
	fin->set_fortran_order();

	fin->get_matrix(tmp.matrix, tmp.num_cols, tmp.num_rows);
	EXPECT_EQ(tmp.num_rows, nfeats);
	EXPECT_EQ(tmp.num_cols, nvecs);

	for (int32_t i=0; i<tmp.num_rows; i++)
	{
		for (int32_t j=0; j<tmp.num_cols; j++)
		{
			EXPECT_EQ(tmp(i, j), a(i, j));
		}
	}
	SG_UNREF(fin);

	// try read in c order
	fin=new CCSVFile("csvfile_test.csv",'r', NULL, '|', '"');
	fin->skip_lines(2);
	fin->set_c_order();

	fin->get_matrix(tmp.matrix, tmp.num_cols, tmp.num_rows);
	EXPECT_EQ(tmp.num_rows, nvecs);
	EXPECT_EQ(tmp.num_cols, nfeats);

	for (int32_t i=0; i<tmp.num_rows; i++)
	{
		for (int32_t j=0; j<tmp.num_cols; j++)
		{
			EXPECT_EQ(tmp(i, j), b(i, j));
		}
	}
	SG_UNREF(fin);
}

TEST(CSVFileTest, write_matrix)
{
	int32_t nvecs=3;
	int32_t nfeats=4;

	float64_t fortran_order_data[]={1.0000, 1.0001, 1.0002, 1.0003,
				1.0004, 1.0005, 1.0006, 1.0007,
				1.0008, 1.0009, 1.0010, 1.0011};

	SGMatrix<float64_t> a=SGMatrix<float64_t>(const_cast<float64_t* >(fortran_order_data), nfeats, nvecs, false);

	CCSVFile* fin;
	CCSVFile* fout;

	SGMatrix<float64_t> tmp(true);

	// try write/read in fortran order
	fout=new CCSVFile("csvfile_test.csv",'w', NULL, '|', '"');
	fout->set_fortran_order();
	fout->set_matrix(fortran_order_data, nvecs, nfeats);
	SG_UNREF(fout);

	fin=new CCSVFile("csvfile_test.csv",'r', NULL, '|', '"');
	fin->set_fortran_order();

	fin->get_matrix(tmp.matrix, tmp.num_cols, tmp.num_rows);
	EXPECT_EQ(tmp.num_rows, nfeats);
	EXPECT_EQ(tmp.num_cols, nvecs);

	for (int32_t i=0; i<tmp.num_rows; i++)
	{
		for (int32_t j=0; j<tmp.num_cols; j++)
		{
			EXPECT_EQ(tmp(i, j), a(i, j));
		}
	}
	SG_UNREF(fin);

	// try write/read in c order
	fout=new CCSVFile("csvfile_test.csv",'w', NULL, '|', '"');
	fout->set_c_order();
	fout->set_matrix(fortran_order_data, nvecs, nfeats);
	SG_UNREF(fout);

	fin=new CCSVFile("csvfile_test.csv",'r', NULL, '|', '"');
	fin->set_c_order();

	fin->get_matrix(tmp.matrix, tmp.num_cols, tmp.num_rows);
	EXPECT_EQ(tmp.num_rows, nfeats);
	EXPECT_EQ(tmp.num_cols, nvecs);

	for (int32_t i=0; i<tmp.num_rows; i++)
	{
		for (int32_t j=0; j<tmp.num_cols; j++)
		{
			EXPECT_EQ(tmp(i, j), a(i, j));
		}
	}
	SG_UNREF(fin);
}
