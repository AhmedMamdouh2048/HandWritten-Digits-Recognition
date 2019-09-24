#include "NeuralNetwork.h"
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
NeuralNetwork::NeuralNetwork(Arguments* A)
{
    Arg = A;

    /*Set the number of CPU cores available*/
    Cores =  thread::hardware_concurrency();

    /*Set the number of dictionaries based on the number of CPU cores with one extra dictionary for non-threaded tasks*/
    FC_Cache = new Mat_Dictionary[Cores + 1];
    FC_Grades = new Mat_Dictionary[Cores + 1];
    if (Arg->NetType == LENET1 || Arg->NetType == other)
    {
        Conv_Cache = new VectVol_Dictionary[Cores + 1];
		Conv_Cache_Mat = new Mat_Dictionary[Cores + 1];
        Conv_Grades = new VectVol_Dictionary[Cores + 1];
        Conv_dbiases = new Mat_Dictionary[Cores + 1];
    }

    /*Set the names of these dictionaries*/
    FC_Parameters.setName("FC_Parameters");
    if (Arg->NetType == LENET1 || Arg->NetType == other)
    {
        Conv_biases.setName("Conv_biases");
        Conv_Weights.setName("Conv_Weights");
    }
    for (int i = 0; i < Cores + 1; i++)
    {
        FC_Cache[i].setName(CharGen("FC_Cache", i));
        FC_Grades[i].setName(CharGen("FC_Grades", i));
        if(Arg->NetType == LENET1 || Arg->NetType == other)
        {
            Conv_Cache[i].setName(CharGen("Conv_Cache",i));
			Conv_Cache_Mat[i].setName(CharGen("ConvCache", i));
            Conv_Grades[i].setName(CharGen("Conv_Grades",i));
            Conv_dbiases[i].setName(CharGen("Conv_dbiases", i));
        }
    }
    if(Arg->NetType == LENET1 || Arg->NetType == other)
    {
        ADAM_dWC.setName("ADAM_dWC");
        ADAM_dbC.setName("ADAM_dbC");
    }

    /*Initialize the required network*/
    switch(Arg->NetType)
    {
    case FC:
        init_FC();
        break;
    case LENET1:
        init_LeNet1();
        break;
    case other:
        init_other();
        break;
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::train()
{
	switch (Arg->NetType)
	{
	case FC:
	    if(Arg->BatchMultiThread)
            train_FC_thread();
        else
            train_FC();
		break;
	case LENET1:
	    if(Arg->BatchMultiThread)
            train_LeNet1_thread();
        else
            train_LeNet1();
		break;
    case other:
        if(Arg->BatchMultiThread)
            train_other_thread();
        else
            train_other();
        break;
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::test(Mode devOrtest)
{
    /*ARGUMENT LIST*/
    Matrix* Y_dev = Arg->Y_dev;
    Matrix* Y_test = Arg->Y_test;
    /*END OF ARGUMENT LIST*/

	clock_t START = clock();

	Matrix* Y_hat = nullptr;
	switch (Arg->NetType)
	{
	case FC:
		test_FC(devOrtest);
		break;
	case LENET1:
		Y_hat = test_LeNet1(devOrtest);
		if(devOrtest == DEV)
            AccuracyTest(Y_dev, Y_hat, "dev");
        else
            AccuracyTest(Y_test, Y_hat, "test");
		FC_Cache[Cores].DeleteThenClear();
		Conv_Cache[Cores].DeleteThenClearObj();
		break;
    case other:
		Y_hat = test_other(devOrtest);
		if(devOrtest == DEV)
            AccuracyTest(Y_dev, Y_hat, "dev");
        else
            AccuracyTest(Y_test, Y_hat, "test");
		FC_Cache[Cores].DeleteThenClear();
		Conv_Cache[Cores].DeleteThenClearObj();
		break;
	}
	clock_t END = clock();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
