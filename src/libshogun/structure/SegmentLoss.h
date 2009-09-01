#ifndef __SEGMENT_LOSS__
#define __SEGMENT_LOSS__

#include "lib/common.h"     
#include "base/SGObject.h"
                   
/** @brief class IntronList */   
class CSegmentLoss : public CSGObject
{                         
        public:
                CSegmentLoss();
                virtual ~CSegmentLoss();

		float32_t get_segment_loss(int32_t from_pos, int32_t to_pos, int32_t segment_id);

		/** set best path segment loss
		 *
		 * @param segment_loss segment loss
		 * @param num_segment_id1 number of segment id1
		 * @param num_segment_id2 number of segment id2
		 */
		void set_segment_loss(float64_t* segment_loss, int32_t m, int32_t n);

		/** set best path segmend ids
		 *
		 * @param segment_ids segment ids
		 */
		void set_segment_ids(CArray<int32_t> *segment_ids);

		void set_segment_mask(CArray<float64_t> *segment_mask);
		
		inline virtual const char* get_name() const { return "SegmentLoss"; }
	protected:             

		/** segment loss 
		 *  two square matrices: 
		 *  one for segment based loss and 
		 *  one for length contribution*/
		CArray3<float64_t> m_segment_loss;
		/** segment IDs */
		CArray<int32_t> m_segment_ids;
		/** segment mask */
		CArray<float64_t> m_segment_mask;
		/** number of different segment types (former: max_a_id)*/
		int32_t m_num_segment_types;
		/** segment loss matrix*/
		CArray2<float32_t> m_segment_loss_matrix;
};
#endif                 

