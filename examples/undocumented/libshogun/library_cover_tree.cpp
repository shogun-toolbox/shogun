#include <shogun/lib/CoverTree.h>
#include <iostream>

using namespace shogun;

class FloatPoint
{
public:
    float64_t* data;
    int32_t point_len;
    
    FloatPoint()
    {
        data = NULL;
        point_len = 0;
    }
    
    FloatPoint(int32_t length)
    {
        data = SG_MALLOC(float64_t, length);
        point_len = length;
    }
    
    /*~FloatPoint()
    {
        SG_FREE(data);
    }*/
    
    static float64_t distance(const FloatPoint &p1, const FloatPoint &p2, float64_t upper_bound)
    {
        float64_t sum = 0.;
        for(int32_t i = 0; i < p1.point_len; i++)
        {
            float64_t d = p1.data[i] - p2.data[i];
            sum += d*d;
        }
        return sqrt(sum);
    }

    static void print(const FloatPoint &p)
    {
        for (int32_t i = 0; i < p.point_len; i++)
            std::cout << p.data[i] << " ";
        std::cout << std::endl;
    }
};

void smallTest()
{
    int32_t numDimensions = 2;
    int32_t numNodes = 10;
    
    v_array<FloatPoint> points;
    v_array<FloatPoint> queries;

	alloc(points, numNodes);
	alloc(queries, 1);

    for(int32_t i = 0; i < numNodes; i++) 
    {		
        FloatPoint a(numDimensions);    
		for(int32_t j = 0; j < numDimensions; j++) 
        {
            a.data[j] = (float64_t)rand()/(float64_t)RAND_MAX;
        } 
        push(points, a);
    }

    std::cout << "Building Cover Tree with " << numNodes << " nodes\n";
    node<FloatPoint> top = batch_create(points);
    std::cout << "Cover tree built.\n";
    
    print(0, top);

    std::cout << "random NN searches beginning...\n";
    FloatPoint a(numDimensions);
    for(int32_t j = 0; j < numDimensions; j++) 
    {
        a.data[j] = (float64_t)rand()/(float64_t)RAND_MAX;
    }
    push(queries, a);
    
    node<FloatPoint> top_query = batch_create(queries);	
	    
	v_array<v_array<FloatPoint> > res;
    k_nearest_neighbor(top,top_query,res,1);

    std::cout << "NN searches done.\n";

    std::cout << "Printing results" << std::endl;
    for (int32_t i = 0; i < res.index(); i++)
    {
        for (int32_t j = 0; j<res[i].index(); j++)
            FloatPoint::print(res[i][j]);
        printf("\n");
    }
    std::cout << "results printed" << std::endl;

    std::cout << "Removing all nodes...\n";
    for(int32_t i = 0; i < numNodes; i++) 
    {
        SG_FREE(points[i].data);
    }
    SG_FREE(points.begin);
    std::cout << "Removal done.\n";
}

int main(int argc, char** argv)
{
	init_shogun();
    
    /* initialize random seed: */
    srand(1);    
    smallTest();
    
	exit_shogun();
	return 0;
}

