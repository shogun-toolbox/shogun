#ifndef __SEGMENT_LOSS__
#define __SEGMENT_LOSS__

#include "lib/common.h"     
#include "base/SGObject.h"
#include "lib/Array.h"
#include "lib/Array2.h"
#include "lib/Array3.h"

                   
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
		void set_segment_ids(CArray<int32_t>* segment_ids);

		void set_segment_mask(CArray<float64_t>* segment_mask);

		void set_num_segment_types(int32_t num_segment_types)
		{
			m_num_segment_types = num_segment_types;
		}
		
		void compute_loss(int32_t* all_pos, int32_t len);

		inline virtual const char* get_name() const { return "SegmentLoss"; }
	protected:             
		/** segment loss matrix*/
		CArray2<float32_t> m_segment_loss_matrix;

		/** segment loss 
		 *  two square matrices: 
		 *  one for segment based loss and 
		 *  one for length contribution*/
		CArray3<float64_t> m_segment_loss;
		/** segment IDs */
		CArray<int32_t>* m_segment_ids;
		/** segment mask */
		CArray<float64_t>* m_segment_mask;
		/** number of different segment types (former: max_a_id)*/
		int32_t m_num_segment_types;
};

inline float32_t CSegmentLoss::get_segment_loss(int32_t from_pos, int32_t to_pos, int32_t segment_id)
{
	float32_t diff_contrib = m_segment_loss_matrix.element(segment_id, from_pos)-m_segment_loss_matrix.element(segment_id, to_pos-1);
	float32_t start_contrib = 0;

	//determine if the loss for the last segment 
	// has to be considered in this segment
	bool add_start_contrib=false;

	ASSERT(to_pos<=m_segment_ids->get_array_size()-1) ;
	if (to_pos == m_segment_ids->get_array_size()-1)
		add_start_contrib = true;
	else if (m_segment_ids->element(to_pos)!=m_segment_ids->element(to_pos+1))
		add_start_contrib = true;

	if (add_start_contrib)
	{
		start_contrib = m_segment_loss.element(segment_id, m_segment_ids->element(to_pos), 0)*m_segment_mask->element(to_pos);
	}	
#ifdef DEBUG
		SG_PRINT("segment_id:%i, from_pos:%i, to_pos:%i\n", segment_id, from_pos, to_pos);
		for (int i=from_pos; i<=to_pos; i++)
			SG_PRINT("%i ", m_segment_ids->element(to_pos));
		SG_PRINT(" \n");
		//SG_PRINT("m_segment_ids->element(from_pos):%i, m_segment_ids->element(to_pos):%i\n", m_segment_ids->element(from_pos), m_segment_ids->element(to_pos));
		SG_PRINT("diff_contrib:%f start_contrib:%f\n", diff_contrib, start_contrib);
#endif	
	return start_contrib + diff_contrib;
}

#endif                 

