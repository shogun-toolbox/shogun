#include "NeuralNets.h"
#include "DataAdapter.h"

using namespace shogun;

int main(int argc, char* argv[])
{
	CNNConfig::Initialize();
	CDataAdapter* data_adapter = new CDataAdapter();
	CNeuralNets nets;	
	nets.SetDataAdapter(data_adapter);
	nets.TrainAll();
	nets.TestAll();

	return 0;
}
