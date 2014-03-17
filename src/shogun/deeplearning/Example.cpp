#include "NeuralNets.h"
#include "DataAdapter.h"

using namespace shogun;

int main(int argc, char* argv[])
{
	NNConfig::Initialize();
	DataAdapter* data_adapter = new DataAdapter();
	NeuralNets nets;	
	nets.SetDataAdapter(data_adapter);
	nets.TrainAll();
	nets.TestAll();

	return 0;
}
