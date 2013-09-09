#include <map>
#include <vector>
#include <string>
#include <istream>
#include <ostream>
#include <fstream>
#include <iterator>
#include <iostream>
#include <libgen.h>

using namespace std;

int main (int argc, char **argv)
{
	cin >> noskipws;

	if (argc < 2)
	{
		cout << "Usage: PATH_TO_TEST_BINARY --gtest_list_tests | ./build_test_cases PATH_TO_TEST_BINARY";
		return 1;
	}

	vector<string > testCases;
	string line;
	string currentTestCase;

	while (getline (cin, line))
	{
		if (line.find (" ") != 0)
			testCases.push_back(line.substr(0, line.size()-1));
	}

	ofstream testfilecmake;
	char *base = basename (argv[1]);
	string gtestName (base);

	testfilecmake.open (string (gtestName + "_test.cmake").c_str (), ios::out | ios::trunc);

	if (testfilecmake.is_open ())
	{
		for (size_t i = 0; i < testCases.size(); i++)
		{
			if (testfilecmake.good ())
			{
				string addTest ("ADD_TEST (unit-");
				string testExec (" \"" + string (argv[1]) + "\"");
				string gTestFilter ("\"--gtest_filter=");
				string endParen (".*\")");

				testfilecmake << addTest << testCases[i] << testExec << gTestFilter << testCases[i] << endParen << endl;
			}
		}

		testfilecmake.close ();
	}

	ifstream CTestTestfile ("CTestTestfile.cmake", ifstream::in);
	bool needsInclude = true;
	line.clear ();

	string includeLine = string ("INCLUDE (") +
	gtestName +
	string ("_test.cmake)");

	if (CTestTestfile.is_open ())
	{
		while (CTestTestfile.good ())
		{
			getline (CTestTestfile, line);

			if (line == includeLine)
			needsInclude = false;
		}
		CTestTestfile.close ();
	}

	if (needsInclude)
	{
		ofstream CTestTestfileW ("CTestTestfile.cmake", ofstream::app | ofstream::out);

		if (CTestTestfileW.is_open ())
		{
			CTestTestfileW << includeLine << endl;
			CTestTestfileW.close ();
		}
	}

	return 0;
}
