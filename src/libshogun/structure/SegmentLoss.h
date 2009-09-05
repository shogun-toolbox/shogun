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
		float32_t get_segment_loss_extend(int32_t from_pos, int32_t to_pos, int32_t segment_id);

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

		void set_segment_mask(CArray<float32_t>* segment_mask);

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
		CArray<float32_t>* m_segment_mask;
		/** number of different segment types (former: max_a_id)*/
		int32_t m_num_segment_types;
		
		bool m_use_loss;
};

inline float32_t CSegmentLoss::get_segment_loss(int32_t from_pos, int32_t to_pos, int32_t segment_id)
{

	/*	int32_t from_pos_shift = from_pos ;
		if (print)
		SG_PRINT("# pos=%i,%i  segment_id=%i, m_segment_ids[from-2]=%i (%1.1f), m_segment_ids[from-1]=%i (%1.1f), m_segment_ids[from]=%i (%1.1f), m_segment_ids[from+1]=%i (%1.1f), \n", 
				 from_pos_shift, to_pos, segment_id, 
				 m_segment_ids->element(from_pos_shift-2),  m_segment_loss_matrix.element(segment_id, from_pos_shift-2)-m_segment_loss_matrix.element(segment_id, to_pos),
				 m_segment_ids->element(from_pos_shift-1), m_segment_loss_matrix.element(segment_id, from_pos_shift-1)-m_segment_loss_matrix.element(segment_id, to_pos),
				 m_segment_ids->element(from_pos_shift), m_segment_loss_matrix.element(segment_id, from_pos_shift)-m_segment_loss_matrix.element(segment_id, to_pos),
				 m_segment_ids->element(from_pos_shift+1),  m_segment_loss_matrix.element(segment_id, from_pos_shift+1)-m_segment_loss_matrix.element(segment_id, to_pos)) ;
	while(1)
	{
		while (m_segment_ids->element(from_pos_shift)==m_segment_ids->element(from_pos_shift+1) && from_pos_shift<to_pos)
			from_pos_shift++ ;
		if (print)
			SG_PRINT("# pos=%i,%i  segment_id=%i, m_segment_ids[from-2]=%i (%1.1f), m_segment_ids[from-1]=%i (%1.1f), m_segment_ids[from]=%i (%1.1f), m_segment_ids[from+1]=%i (%1.1f), \n", 
					 from_pos_shift, to_pos, segment_id, 
					 m_segment_ids->element(from_pos_shift-2),  m_segment_loss_matrix.element(segment_id, from_pos_shift-2)-m_segment_loss_matrix.element(segment_id, to_pos),
					 m_segment_ids->element(from_pos_shift-1), m_segment_loss_matrix.element(segment_id, from_pos_shift-1)-m_segment_loss_matrix.element(segment_id, to_pos),
					 m_segment_ids->element(from_pos_shift), m_segment_loss_matrix.element(segment_id, from_pos_shift)-m_segment_loss_matrix.element(segment_id, to_pos),
					 m_segment_ids->element(from_pos_shift+1),  m_segment_loss_matrix.element(segment_id, from_pos_shift+1)-m_segment_loss_matrix.element(segment_id, to_pos)) ;

		if (from_pos_shift>=to_pos)
		{
			//SG_PRINT("break") ;
			break ;
		}
		else from_pos_shift++ ;
		} 
	if (print)
	SG_PRINT("break\n") ; */

	float32_t diff_contrib = m_segment_loss_matrix.element(segment_id, from_pos)-m_segment_loss_matrix.element(segment_id, to_pos);

	return diff_contrib;
}

inline float32_t CSegmentLoss::get_segment_loss_extend(int32_t from_pos, int32_t to_pos, int32_t segment_id)
{
	int32_t from_pos_shift = from_pos ;

	/*SG_PRINT("segment_id=%i, m_segment_ids[from-2]=%i (%1.1f), m_segment_ids[from-1]=%i (%1.1f), m_segment_ids[from]=%i (%1.1f), m_segment_ids[from+1]=%i (%1.1f), \n", 
			 segment_id, 
			 m_segment_ids->element(from_pos_shift-2),  m_segment_loss_matrix.element(segment_id, from_pos_shift-2)-m_segment_loss_matrix.element(segment_id, to_pos),
			 m_segment_ids->element(from_pos_shift-1), m_segment_loss_matrix.element(segment_id, from_pos_shift-1)-m_segment_loss_matrix.element(segment_id, to_pos),
			 m_segment_ids->element(from_pos_shift), m_segment_loss_matrix.element(segment_id, from_pos_shift)-m_segment_loss_matrix.element(segment_id, to_pos),
			 m_segment_ids->element(from_pos_shift+1),  m_segment_loss_matrix.element(segment_id, from_pos_shift+1)-m_segment_loss_matrix.element(segment_id, to_pos)) ;*/
	
	while (from_pos_shift<to_pos && m_segment_ids->element(from_pos_shift)==m_segment_ids->element(from_pos_shift+1))
		from_pos_shift++ ;

	/*SG_PRINT("segment_id=%i, m_segment_ids[from-2]=%i (%1.1f), m_segment_ids[from-1]=%i (%1.1f), m_segment_ids[from]=%i (%1.1f), m_segment_ids[from+1]=%i (%1.1f), \n", 
			 segment_id, 
			 m_segment_ids->element(from_pos_shift-2),  m_segment_loss_matrix.element(segment_id, from_pos_shift-2)-m_segment_loss_matrix.element(segment_id, to_pos),
			 m_segment_ids->element(from_pos_shift-1), m_segment_loss_matrix.element(segment_id, from_pos_shift-1)-m_segment_loss_matrix.element(segment_id, to_pos),
			 m_segment_ids->element(from_pos_shift), m_segment_loss_matrix.element(segment_id, from_pos_shift)-m_segment_loss_matrix.element(segment_id, to_pos),
			 m_segment_ids->element(from_pos_shift+1),  m_segment_loss_matrix.element(segment_id, from_pos_shift+1)-m_segment_loss_matrix.element(segment_id, to_pos)) ;*/
	
	float32_t diff_contrib = m_segment_loss_matrix.element(segment_id, from_pos_shift)-m_segment_loss_matrix.element(segment_id, to_pos);

	//if (from_pos_shift!=from_pos)
	//	SG_PRINT("shifting from %i to %i, to_pos=%i, loss=%1.1f\n", from_pos, from_pos_shift, to_pos, diff_contrib) ;

	return diff_contrib;
}

#endif                 

