#ifndef _LABELS__H__
#define _LABELS__H__

#include "lib/common.h"

#include <assert.h>

class CLabels
{
	public:
		CLabels(long num_labels);
		CLabels(char* fname);
		~CLabels();

		bool load(char* fname);
		bool save(char* fname);

		/// set/get the labels
		inline bool set_label(long idx, int label)
		{ 
			if (labels && idx<num_labels)
			{
				labels[idx]=label;
				return true;
			}
			else 
				return false;
		}

		inline INT get_label(long idx)
		{
			if (labels && idx<num_labels)
				return labels[idx];
			else
				return -1;
		}

		/// get label vector
		/// caller has to clean up
		INT* get_labels(long &len) ;

		/// get number of labels
		inline int get_num_labels() { return num_labels; }
	protected:
		long num_labels;
		INT* labels;
};
#endif
