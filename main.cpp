#include "DataSet.h"
#include "NeuralNetwork.h"
#include "ConvFeedForward.h"
#define TRAIN_EXAMPLES 60000
#define DEV_EXAMPLES   10000
#define TEST_EXAMPLES  10000
#define ENLARGE_FACT   0
#define ever ;;
using namespace std;
int main()
{
    srand(time(NULL));
	clock_t START = clock();
	Volume  X_2D(TRAIN_EXAMPLES);
	Matrix* Y = new Matrix(10, TRAIN_EXAMPLES);
	Matrix* X_test = new Matrix(784, TEST_EXAMPLES);
	Matrix* Y_test = new Matrix(10, TEST_EXAMPLES);
    const char* dir1  = "F:\\GradProj 2\\dataset\\train-images-idx3-ubyte\\train-images.idx3-ubyte";
	const char* dir2  = "F:\\GradProj 2\\dataset\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte";
	const char* tdir1 = "F:\\GradProj 2\\dataset\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte";
	const char* tdir2 = "F:\\GradProj 2\\dataset\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte";
	get_dataset_2D(X_2D, Y, dir1, dir2, TRAIN_EXAMPLES);
    get_dataset(X_test, Y_test, tdir1, tdir2, TEST_EXAMPLES);
	IntMatrix* X = enlarge1D(X_2D, Y, ENLARGE_FACT);
	X_2D.DELETE();
	clock_t END = clock();

	cout << endl << ">> DataSet Information:" <<endl;
	cout << "Training Images = " << (TRAIN_EXAMPLES* (ENLARGE_FACT + 1)) << endl;
	cout << "Test Images = " << TEST_EXAMPLES << endl;
	cout << "Preprocessing Time = " << (END - START) / CLOCKS_PER_SEC << " Secs " << endl <<endl;


	//------------------------------------------------------------------//
	//--------------------NetWork Architecture--------------------------//
	//------------------------------------------------------------------//
	int numOfLayers = 4;
	layer*  layers = new layer[numOfLayers];
	layers[0].put(784, NONE);
	layers[1].put(600, LEAKYRELU);
	layers[2].put(400, LEAKYRELU);
	layers[3].put(10, SOFTMAX);
	float*  keep_prob = new float[numOfLayers];
	keep_prob[0] = 1;
	keep_prob[1] = 0.6;
	keep_prob[2] = 0.6;
	keep_prob[3] = 1;

	Arguments Arg;
	Arg.NetType = FC;
    Arg.layers = layers;
    Arg.numOfLayers = numOfLayers;
    Arg.X = X;
    Arg.Y = Y;
    Arg.X_dev = nullptr;
    Arg.Y_dev = nullptr;
    Arg.X_test = X_test;
    Arg.Y_test = Y_test;
    Arg.learingRate = 0.03;
    Arg.decayRate = 0.92;
    Arg.numOfEpochs = 1;
    Arg.batchSize = 512;
    Arg.optimizer = ADAM;
    Arg.numPrint = 1;
    Arg.ErrType = CROSS_ENTROPY;
    Arg.regularizationParameter = 0;
    Arg.batchNorm = true;
    Arg.dropout = true;
    Arg.dropConnect = false;
    Arg.keep_prob = keep_prob;
    Arg.BatchMultiThread = false;
    Arg.EB = false;
    Arg.path = "Anything";
    //------------------------------------------------------------------//
    //------------------------------------------------------------------//
    //------------------------------------------------------------------//



    //------------------------------------------------------------------//
    //--------------------Print NetWork Architecture--------------------//
    //------------------------------------------------------------------//
    cout << ">> Training Information: " <<endl;
    cout << "Type Of Network: ";
    switch(Arg.NetType)
    {
        case FC: cout << "Fully Connected" << endl; break;
        case LENET1: cout << "LENET1" << endl; break;
        case other: cout << "Convolution";
    }
    cout << "Optimization Algorithm: ";
    switch(Arg.optimizer)
    {
        case ADAM: cout << "ADAM" << endl; break;
        case GRADIENT_DESCENT: cout << "Gradient Descent" << endl;
    }
    cout << "Cost Function: ";
    switch(Arg.ErrType)
    {
        case CROSS_ENTROPY: cout<< "Cross Entropy" << endl; break;
        case SQAURE_ERROR: cout<< "Square Error" << endl;
    }
    cout << "Learining Rate = " << Arg.learingRate <<endl;
    cout << "Batch Size = " << Arg.batchSize <<endl;
    //------------------------------------------------------------------//
    //------------------------------------------------------------------//
    //------------------------------------------------------------------//



    //------------------------------------------------------------------//
	//-------------------------- Training ------------------------------//
	//------------------------------------------------------------------//
    NeuralNetwork NN(&Arg);
	int i = 0;
	for(ever)
    {
        clock_t start = clock();
        cout << endl <<">> Epoch no. " << ++i << ":"<<endl;
        NN.train();
        NN.test(TEST);
        Arg.learingRate = Arg.learingRate * Arg.decayRate;
        clock_t end = clock();
        double duration_sec = double(end - start) / CLOCKS_PER_SEC;
        cout << "Time = " << duration_sec << endl;
    }
    //------------------------------------------------------------------//
    //------------------------------------------------------------------//
    //------------------------------------------------------------------//

	_getche();
	return 0;
}
