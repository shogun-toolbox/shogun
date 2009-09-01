#include <stdio.h>
#include <string.h>

#include "lib/Mathematics.h"
#include "lib/config.h"
#include "lib/io.h"
#include "structure/SegmentLoss.h"

CSegmentLoss::CSegmentLoss()
:CSGObject()
{

}


void CSegmentLoss::best_path_set_segment_loss(
	float64_t* segment_loss, int32_t m, int32_t n)
{
	// here we need two matrices. Store it in one: 2N x N
	if (2*m!=n)
		SG_ERROR( "segment_loss should be 2 x quadratic matrix: %i!=%i\n", 2*m, n) ;

	if (m!=m_max_a_id+1)
		SG_ERROR( "segment_loss size should match m_max_a_id: %i!=%i\n", m, m_max_a_id+1) ;

	m_segment_loss.set_array(segment_loss, m, n/2, 2, true, true) ;
	/*for (int32_t i=0; i<n; i++)
		for (int32_t j=0; j<n; j++)
		SG_DEBUG( "loss(%i,%i)=%f\n", i,j, m_segment_loss.element(0,i,j)) ;*/
}

void CSegmentLoss::best_path_set_segment_ids_mask(
	int32_t* segment_ids, float64_t* segment_mask, int32_t m)
{
	int32_t max_id = 0;
	for (int32_t i=1;i<m;i++)
		max_id = CMath::max(max_id,segment_ids[i]);
	//SG_PRINT("max_id: %i, m:%i\n",max_id, m); 	
	m_segment_ids.set_array(segment_ids, m, true, true) ;
	m_segment_ids.set_name("m_segment_ids");
	m_segment_mask.set_array(segment_mask, m, true, true) ;
	m_segment_mask.set_name("m_segment_mask");
}
