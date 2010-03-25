
/*
int32_t* CFile::load_int_data(int32_t* target, int64_t& num)
{
	ASSERT(expected_type==F_INT);
	CSimpleFile<int32_t> f(filename, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_int_data(int32_t* src, int64_t num)
{
	ASSERT(expected_type==F_INT);
	CSimpleFile<int32_t> f(filename, file);
	status=f.save(src, num);
	return status;
}

float64_t* CFile::load_real_data(float64_t* target, int64_t& num)
{
	ASSERT(expected_type==F_DREAL);
	CSimpleFile<float64_t> f(filename, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

float32_t* CFile::load_shortreal_data(float32_t* target, int64_t& num)
{
	ASSERT(expected_type==F_SHORTREAL);
	CSimpleFile<float32_t> f(filename, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_real_data(float64_t* src, int64_t num)
{
	ASSERT(expected_type==F_DREAL);
	CSimpleFile<float64_t> f(filename, file);
	status=f.save(src, num);
	return status;
}

bool CFile::save_shortreal_data(float32_t* src, int64_t num)
{
	ASSERT(expected_type==F_SHORTREAL);
	CSimpleFile<float32_t> f(filename, file);
	status=f.save(src, num);
	return status;
}

char* CFile::load_char_data(char* target, int64_t& num)
{
	ASSERT(expected_type==F_CHAR);
	CSimpleFile<char> f(filename, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_char_data(char* src, int64_t num)
{
	ASSERT(expected_type==F_CHAR);
	CSimpleFile<char> f(filename, file);
	status=f.save(src, num);
	return status;
}

uint8_t* CFile::load_byte_data(uint8_t* target, int64_t& num)
{
	ASSERT(expected_type==F_BYTE);
	CSimpleFile<uint8_t> f(filename, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_byte_data(uint8_t* src, int64_t num)
{
	ASSERT(expected_type==F_BYTE);
	CSimpleFile<uint8_t> f(filename, file);
	status=f.save(src, num);
	return status;
}

uint16_t* CFile::load_word_data(uint16_t* target, int64_t& num)
{
	ASSERT(expected_type==F_WORD);
	CSimpleFile<uint16_t> f(filename, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_word_data(uint16_t* src, int64_t num)
{
	ASSERT(expected_type==F_WORD);
	CSimpleFile<uint16_t> f(filename, file);
	status=f.save(src, num);
	return status;
}

int16_t* CFile::load_short_data(int16_t* target, int64_t& num)
{
	ASSERT(expected_type==F_SHORT);
	CSimpleFile<int16_t> f(filename, file);
	target=f.load(target, num);
	status=(target!=NULL);
	return target;
}

bool CFile::save_short_data(int16_t* src, int64_t num)
{
	ASSERT(expected_type==F_SHORT);
	CSimpleFile<int16_t> f(filename, file);
	status=f.save(src, num);
	return status;
}
	int32_t i=0;
	int32_t num_left=get_num_vec_lhs();
	int32_t num_right=rhs->get_num_vectors();
	KERNELCACHE_IDX num_total=num_left*num_right;

	CFile f(fname, 'w', F_DREAL);

    for (int32_t l=0; l< (int32_t) num_left && f.is_ok(); l++)
	{
		for (int32_t r=0; r< (int32_t) num_right && f.is_ok(); r++)
		{
			 if (!(i % (num_total/200+1)))
				SG_PROGRESS(i, 0, num_total-1);

			float64_t k=kernel(l,r);
			f.save_real_data(&k, 1);

			i++;
		}
	}
	SG_DONE();

	if (f.is_ok())
		SG_INFO( "kernel matrix of size %ld x %ld written (filesize: %ld)\n", num_left, num_right, num_total*sizeof(KERNELCACHE_ELEM));

    return (f.is_ok());
*/
