#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/io/File.h>

namespace shogun {

template <class T>
SGSparseMatrix<T>::SGSparseMatrix() : SGReferencedData()
{
	init_data();
}

template <class T>
SGSparseMatrix<T>::SGSparseMatrix(SGSparseVector<T>* vecs, index_t num_feat,
		index_t num_vec, bool ref_counting) :
	SGReferencedData(ref_counting),
	num_vectors(num_vec), num_features(num_feat),
	sparse_matrix(vecs)
{
}

template <class T>
SGSparseMatrix<T>::SGSparseMatrix(index_t num_feat, index_t num_vec, bool ref_counting) :
	SGReferencedData(ref_counting),
	num_vectors(num_vec), num_features(num_feat)
{
	sparse_matrix=SG_MALLOC(SGSparseVector<T>, num_vectors);
}

template <class T>
SGSparseMatrix<T>::SGSparseMatrix(const SGSparseMatrix &orig) : SGReferencedData(orig)
{
	copy_data(orig);
}

template <class T>
SGSparseMatrix<T>::~SGSparseMatrix()
{
	unref();
}

template <> template <>
const SGVector<complex64_t> SGSparseMatrix<complex64_t>::operator*(
	SGVector<float64_t> v) const
{
	SGVector<complex64_t> result(num_vectors);
	REQUIRE(v.vlen==num_features,
		"Dimension mismatch! %d vs %d\n",
		v.vlen, num_features);
	for (index_t i=0; i<num_vectors; ++i)
		result[i]=sparse_matrix[i].dense_dot(v);
	return result;
}

template <> template <>
const SGVector<complex64_t> SGSparseMatrix<complex64_t>::operator*(
	SGVector<int32_t> v) const
{
	SGVector<complex64_t> result(num_vectors);
	REQUIRE(v.vlen==num_features,
		"Dimension mismatch! %d vs %d\n",
		v.vlen, num_features);
	for (index_t i=0; i<num_vectors; ++i)
		result[i]=sparse_matrix[i].dense_dot(v);
	return result;
}

template <> template <>
const SGVector<float64_t> SGSparseMatrix<float64_t>::operator*(
	SGVector<int32_t> v) const
{
	SGVector<float64_t> result(num_vectors);
	REQUIRE(v.vlen==num_features,
		"Dimension mismatch! %d vs %d\n",
		v.vlen, num_features);
	for (index_t i=0; i<num_vectors; ++i)
		result[i]=sparse_matrix[i].dense_dot(v);
	return result;
}

template<class T>
void SGSparseMatrix<T>::load(CFile* loader)
{
	ASSERT(loader)
	unref();

	SG_SET_LOCALE_C;
	loader->get_sparse_matrix(sparse_matrix, num_features, num_vectors);
	SG_RESET_LOCALE;
}

template<>
void SGSparseMatrix<complex64_t>::load(CFile* loader)
{
	SG_SERROR("SGSparseMatrix::load():: Not supported for complex64_t");
}

template<class T>
void SGSparseMatrix<T>::save(CFile* saver)
{
	ASSERT(saver)

	SG_SET_LOCALE_C;
	saver->set_sparse_matrix(sparse_matrix, num_features, num_vectors);
	SG_RESET_LOCALE;
}

template<>
void SGSparseMatrix<complex64_t>::save(CFile* saver)
{
	SG_SERROR("SGSparseMatrix::save():: Not supported for complex64_t");
}

template <class T>
void SGSparseMatrix<T>::copy_data(const SGReferencedData& orig)
{
	sparse_matrix = ((SGSparseMatrix*)(&orig))->sparse_matrix;
	num_vectors = ((SGSparseMatrix*)(&orig))->num_vectors;
	num_features = ((SGSparseMatrix*)(&orig))->num_features;
}

template <class T>
void SGSparseMatrix<T>::init_data()
{
	sparse_matrix = NULL;
	num_vectors = 0;
	num_features = 0;
}

template <class T>
void SGSparseMatrix<T>::free_data()
{
	SG_FREE(sparse_matrix);
	num_vectors = 0;
	num_features = 0;
}

template<class T> CRegressionLabels* SGSparseMatrix<T>::load_svmlight_file(char* fname,
		bool do_sort_features)
{
	CRegressionLabels* lab=NULL;

	size_t blocksize=1024*1024;
	size_t required_blocksize=blocksize;
	uint8_t* dummy=SG_MALLOC(uint8_t, blocksize);
	FILE* f=fopen(fname, "ro");

	if (f)
	{
		free_data();

		SG_SINFO("counting line numbers in file %s\n", fname)
		size_t sz=blocksize;
		size_t block_offs=0;
		size_t old_block_offs=0;
		fseek(f, 0, SEEK_END);
		size_t fsize=ftell(f);
		rewind(f);

		while (sz == blocksize)
		{
			sz=fread(dummy, sizeof(uint8_t), blocksize, f);
			for (size_t i=0; i<sz; i++)
			{
				block_offs++;
				if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))
				{
					num_vectors++;
					required_blocksize=CMath::max(required_blocksize, block_offs-old_block_offs+1);
					old_block_offs=block_offs;
				}
			}
			SG_SPROGRESS(block_offs, 0, fsize, 1, "COUNTING:\t")
		}

		SG_SINFO("found %d feature vectors\n", num_vectors)
		SG_FREE(dummy);
		blocksize=required_blocksize;
		dummy = SG_MALLOC(uint8_t, blocksize+1); //allow setting of '\0' at EOL

		lab=new CRegressionLabels(num_vectors);
		sparse_matrix=SG_MALLOC(SGSparseVector<T>, num_vectors);
		rewind(f);
		sz=blocksize;
		int32_t lines=0;
		while (sz == blocksize)
		{
			sz=fread(dummy, sizeof(uint8_t), blocksize, f);

			size_t old_sz=0;
			for (size_t i=0; i<sz; i++)
			{
				if (i==sz-1 && dummy[i]!='\n' && sz==blocksize)
				{
					size_t len=i-old_sz+1;
					uint8_t* data=&dummy[old_sz];

					for (size_t j=0; j<len; j++)
						dummy[j]=data[j];

					sz=fread(dummy+len, sizeof(uint8_t), blocksize-len, f);
					i=0;
					old_sz=0;
					sz+=len;
				}

				if (dummy[i]=='\n' || (i==sz-1 && sz<blocksize))
				{

					size_t len=i-old_sz;
					uint8_t* data=&dummy[old_sz];

					int32_t dims=0;
					for (size_t j=0; j<len; j++)
					{
						if (data[j]==':')
							dims++;
					}

					if (dims<=0)
					{
						SG_SERROR("Error in line %d - number of"
								" dimensions is %d line is %d characters"
								" long\n line_content:'%.*s'\n", lines,
								dims, len, len, (const char*) data);
					}

					SGSparseVectorEntry<T>* feat=SG_MALLOC(SGSparseVectorEntry<T>, dims);
					size_t j=0;
					for (; j<len; j++)
					{
						if (data[j]==' ')
						{
							data[j]='\0';

							lab->set_label(lines, atof((const char*) data));
							break;
						}
					}

					int32_t d=0;
					j++;
					uint8_t* start=&data[j];
					for (; j<len; j++)
					{
						if (data[j]==':')
						{
							data[j]='\0';

							feat[d].feat_index=(int32_t) atoi((const char*) start)-1;
							num_features=CMath::max(num_features, feat[d].feat_index+1);

							j++;
							start=&data[j];
							for (; j<len; j++)
							{
								if (data[j]==' ' || data[j]=='\n')
								{
									data[j]='\0';
									feat[d].entry=(T) atof((const char*) start);
									d++;
									break;
								}
							}

							if (j==len)
							{
								data[j]='\0';
								feat[dims-1].entry=(T) atof((const char*) start);
							}

							j++;
							start=&data[j];
						}
					}

					sparse_matrix[lines].num_feat_entries=dims;
					sparse_matrix[lines].features=feat;

					old_sz=i+1;
					lines++;
					SG_SPROGRESS(lines, 0, num_vectors, 1, "LOADING:\t")
				}
			}
		}
		SG_SINFO("file successfully read\n")
		fclose(f);
	}

	SG_FREE(dummy);

	if (do_sort_features)
		sort_features();

	return lab;
}

template<class T> SGSparseMatrix<T> SGSparseMatrix<T>::get_transposed()
{
	SGSparseMatrix<T> sfm(num_vectors, num_features);

	int32_t* hist=SG_CALLOC(int32_t, num_features);

	// count the lengths of future feature vectors
	for (int32_t v=0; v<num_vectors; v++)
	{
		SGSparseVector<T> sv=sparse_matrix[v];

		for (int32_t i=0; i<sv.num_feat_entries; i++)
			hist[sv.features[i].feat_index]++;
	}

	for (int32_t v=0; v<num_features; v++)
		sfm[v]=SGSparseVector<T>(hist[v]);

	SG_FREE(hist);

	int32_t* index=SG_CALLOC(int32_t, num_vectors);

	// fill future feature vectors with content
	for (int32_t v=0; v<num_vectors; v++)
	{
		SGSparseVector<T> sv=sparse_matrix[v];

		for (int32_t i=0; i<sv.num_feat_entries; i++)
		{
			int32_t vidx=sv.features[i].feat_index;
			int32_t fidx=v;
			sfm[vidx].features[index[vidx]].feat_index=fidx;
			sfm[vidx].features[index[vidx]].entry=sv.features[i].entry;
			index[vidx]++;
		}
	}

	SG_FREE(index);
	return sfm;
}


template<class T> void SGSparseMatrix<T>::sort_features()
{
	for (int32_t i=0; i<num_vectors; i++)
	{
		sparse_matrix[i].sort_features();
	}
}

template<class T> bool SGSparseMatrix<T>::write_svmlight_file(char* fname,
		CRegressionLabels* label)
{
	ASSERT(label)
	int32_t num=label->get_num_labels();
	ASSERT(num>0)
	ASSERT(num==num_vectors)

	FILE* f=fopen(fname, "wb");

	if (f)
	{
		for (int32_t i=0; i<num; i++)
		{
			fprintf(f, "%d ", (int32_t) label->get_int_label(i));

			SGSparseVectorEntry<T>* vec = sparse_matrix[i].features;
			int32_t num_feat = sparse_matrix[i].num_feat_entries;

			for (int32_t j=0; j<num_feat; j++)
			{
				if (j<num_feat-1)
					fprintf(f, "%d:%f ", (int32_t) vec[j].feat_index+1, (double) vec[j].entry);
				else
					fprintf(f, "%d:%f\n", (int32_t) vec[j].feat_index+1, (double) vec[j].entry);
			}
		}

		fclose(f);
		return true;
	}
	return false;
}

template <> bool SGSparseMatrix<complex64_t>::write_svmlight_file(char* fname, CRegressionLabels* label) { return false; }

template<> CRegressionLabels* SGSparseMatrix<complex64_t>::load_svmlight_file(char* fname, bool do_sort_features) { return NULL; }

template class SGSparseMatrix<bool>;
template class SGSparseMatrix<char>;
template class SGSparseMatrix<int8_t>;
template class SGSparseMatrix<uint8_t>;
template class SGSparseMatrix<int16_t>;
template class SGSparseMatrix<uint16_t>;
template class SGSparseMatrix<int32_t>;
template class SGSparseMatrix<uint32_t>;
template class SGSparseMatrix<int64_t>;
template class SGSparseMatrix<uint64_t>;
template class SGSparseMatrix<float32_t>;
template class SGSparseMatrix<float64_t>;
template class SGSparseMatrix<floatmax_t>;
template class SGSparseMatrix<complex64_t>;
}
