/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Berlin Institute of Technology
 */

#ifndef _CSTRINGFILEFEATURES__H__
#define _CSTRINGFILEFEATURES__H__

#include <shogun/features/StringFeatures.h>
#include <shogun/features/Alphabet.h>
#include <shogun/io/MemoryMappedFile.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>

namespace shogun
{
class CAlphabet;
template <class T> class CMemoryMappedFile;

/** @brief File based string features.
 *
 * StringFeatures that are file based. Underneath memory mapped files are used.
 * Derived from CStringFeatures thus transparently enabling all of the
 * StringFeature functionality.
 *
 * Supported file format contains one string per line, lines of variable length
 * are supported and must be separated by '\n'.
 */
template <class ST> class CStringFileFeatures : public CStringFeatures<ST>
{
	public:

	/** default constructor
	 *
	 */
	CStringFileFeatures() : CStringFeatures<ST>(), file(NULL)
	{
	}

	/** constructor
	 *
	 * @param fname filename of the file containing line based features
	 * @param alpha alphabet (type) to use for string features
	 */
	CStringFileFeatures(const char* fname, EAlphabet alpha)
	: CStringFeatures<ST>(alpha)
	{
		file = new CMemoryMappedFile<ST>(fname);
		fetch_meta_info_from_file();
	}

	/** default destructor
	 *
	 */
	virtual ~CStringFileFeatures()
	{
		SG_UNREF(file);
		CStringFileFeatures<ST>::cleanup();
	}

	protected:
	/** get next line from file
	 *
	 * The returned line may be modfied in case the file was opened
	 * read/write. It is otherwise read-only.
	 *
	 * @param len length of line (returned via reference)
	 * @param offs offset to be passed for reading next line, should be 0
	 * 			initially (returned via reference)
	 * @param line_nr used to indicate errors (returned as reference should be 0
	 * 			initially)
	 * @param file_length total length of the file (for error checking)
	 *
	 * @return line (NOT ZERO TERMINATED)
	 */
	ST* get_line(uint64_t& len, uint64_t& offs, int32_t& line_nr, uint64_t file_length)
	{
		ST* s = file->get_map();
		for (uint64_t i=offs; i<file_length; i++)
		{
			ST c=s[i];

			if (c == '\n')
			{
				ST* line=&s[offs];
				len=i-offs;
				offs=i+1;
				line_nr++;
				return line;
			}
			else
			{
				if (!CStringFeatures<ST>::alphabet->is_valid((uint8_t) c))
				{
					CStringFileFeatures<ST>::cleanup();
					CStringFeatures<ST>::SG_ERROR("Invalid character (%c) in line %d\n", c, line_nr);
				}
			}
		}

		len=0;
		offs=file_length;
		return NULL;
	}

	/** cleanup string features */
	virtual void cleanup()
	{
		CStringFeatures<ST>::num_vectors=0;
		SG_FREE(CStringFeatures<ST>::features);
		SG_FREE(CStringFeatures<ST>::symbol_mask_table);
		CStringFeatures<ST>::features=NULL;
		CStringFeatures<ST>::symbol_mask_table=NULL;

		/* start with a fresh alphabet, but instead of emptying the histogram
		 * create a new object (to leave the alphabet object alone if it is used
		 * by others)
		 */
		CAlphabet* alpha=new CAlphabet(CStringFeatures<ST>::alphabet->get_alphabet());
		SG_UNREF(CStringFeatures<ST>::alphabet);
		CStringFeatures<ST>::alphabet=alpha;
		SG_REF(CStringFeatures<ST>::alphabet);
	}

    /** cleanup a single feature vector */
    virtual void cleanup_feature_vector(int32_t num)
    {
        CStringFeatures<ST>::SG_ERROR("Cleaning single feature vector not"
                "supported by StringFileFeatures\n");
    }

	/** obtain meta information from file
	 * 
	 * i.e., determine number of strings and their lengths
	 */
	void fetch_meta_info_from_file(int32_t granularity=1048576)
	{
		CStringFileFeatures<ST>::cleanup();
		uint64_t file_size=file->get_size();
		ASSERT(granularity>=1);
		ASSERT(CStringFeatures<ST>::alphabet);

		uint64_t buffer_size=granularity;
		CStringFeatures<ST>::features=SG_MALLOCX(SGString<ST>, buffer_size);

		uint64_t offs=0;
		uint64_t len=0;
		CStringFeatures<ST>::max_string_length=0;
		CStringFeatures<ST>::num_vectors=0;

		while (true)
		{
			ST* line=get_line(len, offs, CStringFeatures<ST>::num_vectors, file_size);

			if (line)
			{
				if (CStringFeatures<ST>::num_vectors>buffer_size)
				{
					CMath::resize(CStringFeatures<ST>::features, buffer_size, buffer_size+granularity);
					buffer_size+=granularity;
				}

				CStringFeatures<ST>::features[CStringFeatures<ST>::num_vectors-1].string=line;
				CStringFeatures<ST>::features[CStringFeatures<ST>::num_vectors-1].length=len;
				CStringFeatures<ST>::max_string_length=CMath::max(CStringFeatures<ST>::max_string_length, (int32_t) len);
			}
			else
				break;
		}

		CStringFeatures<ST>::SG_INFO("number of strings:%d\n", CStringFeatures<ST>::num_vectors);
		CStringFeatures<ST>::SG_INFO("maximum string length:%d\n", CStringFeatures<ST>::max_string_length);
		CStringFeatures<ST>::SG_INFO("max_value_in_histogram:%d\n", CStringFeatures<ST>::alphabet->get_max_value_in_histogram());
		CStringFeatures<ST>::SG_INFO("num_symbols_in_histogram:%d\n", CStringFeatures<ST>::alphabet->get_num_symbols_in_histogram());

		if (!CStringFeatures<ST>::alphabet->check_alphabet_size() || !CStringFeatures<ST>::alphabet->check_alphabet())
			CStringFileFeatures<ST>::cleanup();

		CMath::resize(CStringFeatures<ST>::features, buffer_size, CStringFeatures<ST>::num_vectors);
	}


	protected:
	/** memory mapped file*/
	CMemoryMappedFile<ST>* file;
};
}
#endif // _CSTRINGFILEFEATURES__H__
