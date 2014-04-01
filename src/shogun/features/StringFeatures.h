/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CSTRINGFEATURES__H__
#define _CSTRINGFEATURES__H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/lib/Cache.h>
#include <shogun/lib/DynamicArray.h>
#include <shogun/lib/Compressor.h>
#include <shogun/io/File.h>

#include <shogun/features/Features.h>
#include <shogun/features/Alphabet.h>
#include <shogun/lib/SGString.h>

namespace shogun
{
class CAlphabet;
template <class T> class CDynamicArray;
class CFile;
template <class T> class SGString;
template <class T> class SGStringList;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct SSKDoubleFeature
{
	int feature1;
	int feature2;
	int group;
};

struct SSKTripleFeature
{
	int feature1;
	int feature2;
	int feature3;
	int group;
};
#endif

/** @brief Template class StringFeatures implements a list of strings.
 *
 * As this class is a template the underlying storage type is quite arbitrary and
 * not limited to character strings, but could also be sequences of floating
 * point numbers etc. Strings differ from matrices (cf. CDenseFeatures) in a
 * way that the dimensionality of the feature vectors (i.e. the strings) is not
 * fixed; it may vary between strings.
 *
 * Most string kernels require StringFeatures but a number of them actually
 * requires strings to have same length.
 *
 * When preprocessors are attached to string features they may shorten the
 * string, but are not allowed to return strings longer than max_string_length,
 * as some algorithms depend on this.
 *
 * Also note that string features cannot currently be computed on-the-fly.
 *
 * (Partly) subset access is supported for this feature type.
 * Simple use the (inherited) add_subset(), remove_subset() functions.
 * If done, all calls that work with features are translated to the subset.
 * See comments to find out whether it is supported for that method.
 * See also CFeatures class documentation
 */
template <class ST> class CStringFeatures : public CFeatures
{
	public:
		/** default constructor */
		CStringFeatures();

		/** constructor
		 *
		 * @param alpha alphabet (type) to use for string features
		 */
		CStringFeatures(EAlphabet alpha);

		/** constructor
		 *
		 * @param string_list
		 * @param alpha alphabet (type) to use for string features
		 */
		CStringFeatures(SGStringList<ST> string_list, EAlphabet alpha);

		/** constructor
		 *
		 * @param string_list
		 * @param alpha an actual alphabet
		 */
		CStringFeatures(SGStringList<ST> string_list, CAlphabet* alpha);

		/** constructor
		 *
		 * @param alpha alphabet to use for string features
		 */
		CStringFeatures(CAlphabet* alpha);

		/** copy constructor
		 *
		 * @param orig features to copy
		 */
		CStringFeatures(const CStringFeatures& orig);

		/** constructor
		 *
		 * @param loader File object via which to load data
		 * @param alpha alphabet (type) to use for string features
		 */
		CStringFeatures(CFile* loader, EAlphabet alpha=DNA);

		/** destructor */
		virtual ~CStringFeatures();

		/** cleanup string features.
		 *
		 * removes any subset before
		 *
		 */
		virtual void cleanup();

		/** cleanup a single feature vector
		 *
		 * possible with subset
		 *
		 * @param num number of the vector
		 */
		virtual void cleanup_feature_vector(int32_t num);

		/** cleanup multiple feature vectors
		 *
		 * possible with subset
		 *
		 * @param start index of first vector to be cleaned
		 * @param stop index of the last vector to be cleaned
		 */
		virtual void cleanup_feature_vectors(int32_t start, int32_t stop);

		/** get feature class
		 *
		 * @return feature class STRING
		 */
		virtual EFeatureClass get_feature_class() const;

		/** get feature type
		 *
		 * @return templated feature type
		 */
		virtual EFeatureType get_feature_type() const;

		/** get alphabet used in string features
		 *
		 * @return alphabet
		 */
		CAlphabet* get_alphabet();

		/** duplicate feature object
		 *
		 * @return feature object
		 */
		virtual CFeatures* duplicate() const;

		/** get string for selected example num
		 *
		 * possible with subset
		 *
		 * @param num index of the string
		 * @return the selected string
		 */
		SGVector<ST> get_feature_vector(int32_t num);

		/** set string for selected example num
		 *
		 * not possible with subset
		 *
		 * @param vector string to set
		 * @param num index of the string
		 */
		void set_feature_vector(SGVector<ST> vector, int32_t num);

		/** call this to preprocess string features upon call to get_feature_vector */
		void enable_on_the_fly_preprocessing();

		/** call this to disable on the fly feature preprocessing upon call to
		 * get_feature_vector. Useful when you manually apply preprocessors.
		 */
		void disable_on_the_fly_preprocessing();

		/** get feature vector for sample num
		 *
		 * possible with subset
		 *
		 * @param num index of feature vector
		 * @param len length is returned by reference
		 * @param dofree whether returned vector must be freed by
		 * caller via free_feature_vector
		 * @return feature vector for sample num
		 */
		ST* get_feature_vector(int32_t num, int32_t& len, bool& dofree);

		/** get a transposed copy of the features
		 *
		 * possible with subset
		 *
		 * @return transposed copy
		 */
		CStringFeatures<ST>* get_transposed();

		/** compute and return the transpose of string features matrix
		 * which will be prepocessed.
		 * num_feat, num_vectors are returned by reference
		 * caller has to clean up
		 *
		 * note that strings all have to have same length
		 *
		 * possible with subset
		 *
		 * @param num_feat number of features in matrix
		 * @param num_vec number of vectors in matrix
		 * @return transposed string features
		 */
		SGString<ST>* get_transposed(int32_t &num_feat, int32_t &num_vec);

		/** free feature vector
		 *
		 * possible with subset
		 *
		 * @param feat_vec feature vector to free
		 * @param num index in feature cache, possibly from subset
		 * @param dofree if vector should be really deleted
		 */
		void free_feature_vector(ST* feat_vec, int32_t num, bool dofree);

		/** free feature vector
		 *
		 * possible with subset
		 *
		 * @param feat_vec feature vector to free
		 * @param num index in feature cache, possibly from subset
		 */
		void free_feature_vector(SGVector<ST> feat_vec, int32_t num);

		/** get feature
		 *
		 * possible with subset
		 *
		 * @param vec_num which vector
		 * @param feat_num which feature, possibly from subset
		 * @return feature
		 */
		virtual ST get_feature(int32_t vec_num, int32_t feat_num);

		/** get vector length
		 *
		 * possible with subset
		 *
		 * @param vec_num which vector, possibly from subset
		 * @return length of vector
		 */
		virtual int32_t get_vector_length(int32_t vec_num);

		/** get maximum vector length
		 *
		 * this one is updated when a subset is set
		 *
		 * @return maximum vector/string length
		 */
		virtual int32_t get_max_vector_length();

		/** @return number of vectors, possibly of subset */
		virtual int32_t get_num_vectors() const;

		/** get number of symbols
		 *
		 * Note: floatmax_t sounds weird, but LONG is not long enough
		 *
		 * @return number of symbols
		 */
		floatmax_t get_num_symbols();

		/** get maximum number of symbols
		 *
		 * Note: floatmax_t sounds weird, but int64_t is not long enough (and
		 * there is no int128_t type)
		 *
		 * @return maximum number of symbols
		 */
		floatmax_t get_max_num_symbols();

		// these functions are necessary to find out about a former conversion process

		/** number of symbols before higher order mapping
		 *
		 * @return original number of symbols
		 */
		floatmax_t get_original_num_symbols();

		/** order used for higher order mapping
		 *
		 * @return order
		 */
		int32_t get_order();

		/** a higher order mapped symbol will be shaped such that the symbols
		 * specified by bits in the mask will be returned.
		 *
		 * @param symbol symbol to mask
		 * @param mask mask to apply
		 * @return masked symbol
		 */
		ST get_masked_symbols(ST symbol, uint8_t mask);

		/** shift offset to the left by amount
		 *
		 * @param offset offset to shift
		 * @param amount amount to shift the offset
		 * @return shifted offset
		 */
		ST shift_offset(ST offset, int32_t amount);

		/** shift symbol to the right by amount (taking care of custom symbol sizes)
		 *
		 * @param symbol symbol to shift
		 * @param amount amount to shift the symbol
		 * @return shifted symbol
		 */
		ST shift_symbol(ST symbol, int32_t amount);

		/** load features from file
		 *
		 * @param loader File object via which to load data
		 */
		virtual void load(CFile* loader);

		/** load ascii line-based string features from file.
		 *
		 * any subset is removed before
		 *
		 * @param fname filename to load from
		 * @param remap_to_bin if translation to other binary alphabet
		 * should be performed
		 * @param ascii_alphabet src alphabet
		 * @param binary_alphabet alphabet to translate to
		 */
		void load_ascii_file(char* fname, bool remap_to_bin=true,
				EAlphabet ascii_alphabet=DNA, EAlphabet binary_alphabet=RAWDNA);

		/** load fasta file as string features
		 *
		 * any subset is removed before
		 *
		 * @param fname filename to load from
		 * @param ignore_invalid if set to true, characters other than A,C,G,T are converted to A
		 * @return if loading was successful
		 */
		bool load_fasta_file(const char* fname, bool ignore_invalid=false);

		/** load fastq file as string features
		 *
		 * removes subset beforehand
		 *
		 * @param fname filename to load from
		 * @param ignore_invalid if set to true, characters other than A,C,G,T are converted to A
		 * @param bitremap_in_single_string if set to true, do binary embedding of symbols
		 * @return if loading was successful
		 */
		bool load_fastq_file(const char* fname,
				bool ignore_invalid=false, bool bitremap_in_single_string=false);

		/** load features from directory
		 *
		 *removes subset before
		 *
		 * @param dirname directory name to load from
		 * @return if loading was successful
		 */
		bool load_from_directory(char* dirname);

		/** set features
		 *
		 * not possible with subset
		 *
		 */
        void set_features(SGStringList<ST> feats);

		/** set features
		 *
		 * not possible with subset
		 *
		 * @param p_features new features
		 * @param p_num_vectors number of vectors
		 * @param p_max_string_length maximum string length
		 * @return if setting was successful
		 */
		bool set_features(SGString<ST>* p_features, int32_t p_num_vectors,
				int32_t p_max_string_length);

		/** append features
		 * If the given string features have a subset, only this will be copied
		 *
		 * not possible with subset
		 *
		 * @param sf features to append
		 * @return if setting was successful
		 */
		bool append_features(CStringFeatures<ST>* sf);

		/** append features
		 *
		 *  not possible with subset
		 *
		 * @param p_features features to append
		 * @param p_num_vectors number of vectors
		 * @param p_max_string_length maximum string length
		 *
		 * note that p_features will be SG_FREE()'d on success
		 *
		 * @return if setting was successful
		 */
		bool append_features(SGString<ST>* p_features, int32_t p_num_vectors,
				int32_t p_max_string_length);

		/** get_features
		 * @return features
		 */
        SGStringList<ST> get_features();

		/** get_features
		 *
		 * not possible with subset
		 *
		 * @param num_str number of strings (returned)
		 * @param max_str_len maximal string length (returned)
		 * @return string features
		 */
		virtual SGString<ST>* get_features(int32_t& num_str, int32_t& max_str_len);

		/** copy_features
		 *
		 * possible with subset
		 *
		 * @param num_str number of strings (returned)
		 * @param max_str_len maximal string length (returned)
		 * @return string features
		 */
		virtual SGString<ST>* copy_features(int32_t& num_str, int32_t& max_str_len);

		/** get_features  (swig compatible)
		 *
		 * possible with subset
		 *
		 * @param dst string features (returned)
		 * @param num_str number of strings (returned)
		 */
		virtual void get_features(SGString<ST>** dst, int32_t* num_str);

		/** save features to file
		 *
		 * not possible with subset
		 *
		 * @param writer File object via which to save data
		 */
		virtual void save(CFile* writer);

		/** load compressed features from file
		 *
		 * any subset is removed before
		 *
		 * @param src filename to load from
		 * @param decompress whether to decompress on loading
		 * @return if loading was successful
		 */
		virtual bool load_compressed(char* src, bool decompress);

		/** save compressed features to file
		 *
		 * not possible with subset
		 *
		 * @param dest filename to save to
		 * @param compression compressor to use
		 * @param level compression level to use (1-9)
		 * @return if saving was successful
		 */
		virtual bool save_compressed(char* dest, E_COMPRESSION_TYPE compression, int level);

		/** apply preprocessor
		 *
		 * @param force_preprocessing if preprocssing shall be forced
		 * @return if applying was successful
		 */
		virtual bool apply_preprocessor(bool force_preprocessing=false);

		/** slides a window of size window_size over the current single string
		 * step_size is the amount by which the window is shifted.
		 * creates (string_len-window_size)/step_size many feature obj
		 * if skip is nonzero, skip the first 'skip' characters of each string
		 *
		 * not implemented for subset
		 *
		 * @param window_size window size
		 * @param step_size step size
		 * @param skip skip
		 * @return something inty
		 */
		int32_t obtain_by_sliding_window(int32_t window_size, int32_t step_size, int32_t skip=0);

		/** extracts windows of size window_size from first string
		 * using the positions in list
		 *
		 * not implemented for subset
		 *
		 * @param window_size window size
		 * @param positions positions
		 * @param skip skip
		 * @return something inty
		 */
		int32_t obtain_by_position_list(int32_t window_size, CDynamicArray<int32_t>* positions,
				int32_t skip=0);

		/** obtain string features from char features
		 *
		 * wrapper for template method
		 *
		 * any subset is removed before, subset of parameter sf is possible
		 *
		 * @param sf string features
		 * @param start start
		 * @param p_order order
		 * @param gap gap
		 * @param rev reverse
		 * @return if obtaining was successful
		 */
		bool obtain_from_char(CStringFeatures<char>* sf, int32_t start,
				int32_t p_order, int32_t gap, bool rev);

		/** template obtain from char features
		 *
		 * any subset is removed before, subset of parameter sf is possible
		 *
		 * @param sf string features
		 * @param start start
		 * @param p_order order
		 * @param gap gap
		 * @param rev reverse
		 * @return if obtaining was successful
		 */
		template <class CT>
			bool obtain_from_char_features(CStringFeatures<CT>* sf, int32_t start,
					int32_t p_order, int32_t gap, bool rev);

		/** check if length of each vector in this feature object equals the
		 * given length. if existant, only subset is checked
		 *
		 * possible for subset
		 *
		 * @param len vector length to check against
		 * @return if length of each vector in this feature object equals the
		 * given length.
		 */
		bool have_same_length(int32_t len=-1);

		/** embed string features in bit representation in-place
		 *
		 * not implemented for subset
		 *
		 */
		void embed_features(int32_t p_order);

		/** compute symbol mask table
		 *
		 * required to access bit-based symbols
		 *
		 * not implemented for subset
		 */
		void compute_symbol_mask_table(int64_t max_val);

		/** remap bit-based word to character sequence
		 *
		 * @param word word to remap
		 * @param seq sequence of size len that remapped characters are written to
		 * @param len length of sequence and word
		 */
		void unembed_word(ST word, uint8_t* seq, int32_t len);

		/** embed a single word
		 *
		 * @param seq sequence of size len in a bitfield
		 * @param len
		 */
		ST embed_word(ST* seq, int32_t len);

		/** determine new maximum string length
		 *
		 * possible with subset
		 */
		void determine_maximum_string_length();

		/** get a zero terminated copy of the string
		 *
		 * @param str the string to copy
		 * @return zero terminated copy of str
		 *
		 * note that this function is only sensible for character strings
		 */
		static ST* get_zero_terminated_string_copy(SGString<ST> str);

		/** set feature vector for sample num
		 *
		 * possible with subset
		 *
		 * @param num index of feature vector
		 * @param string string with the feature vector's content
		 * @param len length of the string
		 */
		virtual void set_feature_vector(int32_t num, ST* string, int32_t len);

		/** compute histogram over strings
		 *
		 * possible with subset
		 */
		virtual void get_histogram(float64_t** hist, int32_t* rows, int32_t* cols,
				bool normalize=true);

		/** create some random strings based on normalized histogram
		 *
		 * not possible with subset
		 */
		virtual void create_random(float64_t* hist, int32_t rows, int32_t cols,
				int32_t num_vec);

		/** Creates a new CFeatures instance containing copies of the elements
		 * which are specified by the provided indices.
		 *
		 * possible with subset
		 *
		 * @param indices indices of feature elements to copy
		 * @return new CFeatures instance with copies of feature data
		 */
		virtual CFeatures* copy_subset(SGVector<index_t> indices);

		/** @return object name */
		virtual const char* get_name() const { return "StringFeatures"; }

		/** post method when subset is changed */
		virtual void subset_changed_post();

	protected:
		/** compute feature vector for sample num
		 * if target is set the vector is written to target
		 * len is returned by reference
		 *
		 * possible with subset
		 *
		 * @param num which vector
		 * @param len length of vector
		 * @return feature vector
		 */
		virtual ST* compute_feature_vector(int32_t num, int32_t& len);

	private:
		void init();

	protected:
		/** alphabet */
		CAlphabet* alphabet;

		/** number of string vectors (for subset, is updated) */
		int32_t num_vectors;

		/** this contains the array of features */
		SGString<ST>* features;

		/** true when single string / created by sliding window */
		ST* single_string;

		/// length of prior single string
		int32_t length_of_single_string;

		/** length of longest string (for subset, is updated) */
		int32_t max_string_length;

		/// number of used symbols
		floatmax_t num_symbols;

		/// original number of used symbols (before higher order mapping)
		floatmax_t original_num_symbols;

		/// order used in higher order mapping
		int32_t order;

		/// order used in higher order mapping
		ST* symbol_mask_table;

		/// order used in higher order mapping
		int32_t symbol_mask_table_len;

		/// preprocess on-the-fly?
		bool preprocess_on_get;

		/** feature cache */
		CCache<ST>* feature_cache;
};
}
#endif // _CSTRINGFEATURES__H__
