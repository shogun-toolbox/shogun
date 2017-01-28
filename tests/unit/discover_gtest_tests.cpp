#include <map>
#include <vector>
#include <string>
#include <istream>
#include <ostream>
#include <fstream>
#include <iterator>
#include <iostream>
#ifndef _WIN32
#include <libgen.h>
#else
#include <regex>
#endif

#if _MSC_VER >= 1600
#include <algorithm>

std::string basename(const std::string& pathname)
{
	return { std::find_if(pathname.rbegin(), pathname.rend(),
		[](char c) { return c == '/' || c == '\\'; }).base(),
		pathname.end() };
}
#endif

using namespace std;

int main (int argc, char **argv)
{
	cin >> noskipws;

	if (argc < 2)
	{
		cout << "Usage: PATH_TO_TEST_BINARY --gtest_list_tests | ./discover_gtest_tests PATH_TO_TEST_BINARY";
		return 1;
	}

	vector<string > testCases;
	string line;
	string currentTestCase;

	while (getline (cin, line))
	{
		if ((line.find (" ") != 0) && (line[0] != '0'))
			testCases.push_back(line.substr(0, line.rfind('.')));
	}

	ofstream testfilecmake;
#ifdef _MSC_VER
	string gtestName = basename(argv[1]);
#else
	char *base = basename (argv[1]);
	string gtestName (base);
#endif

	testfilecmake.open (string (gtestName + "_test.cmake").c_str (), ios::out | ios::trunc);

	if (testfilecmake.is_open ())
	{
		for (size_t i = 0; i < testCases.size(); i++)
		{
			if (testfilecmake.good ())
			{
				string addTest ("ADD_TEST (unit-");
				string testExec (" \"" + string (argv[1]) + "\"");
#ifdef _WIN32
				// escape backslash in path
				std::regex e ("(\\\\)");
				testExec = std::regex_replace (testExec, e, "\\\\");
#endif
				string gTestFilter (" \"--gtest_filter=");
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
