#include "NeuralNetwork.h"

#define WC1_NoOfFilters 4
#define WC1_Channels 1
#define WC1_f 5
#define WC2_NoOfFilters 12
#define WC2_Channels 4
#define WC2_f 5

//***********************************************************************************************************/
// 4D discription for a 4D element A(m,nc,nh,nw):                                                           //
// 1- A is a vector of volumes that has a size m                                                            //
// 2- A[i] is a volume with nc channels, it represents the activations of some layer for the ith example    //
// 3- A[i][j] is a pointer to the jth Matrix with nh hight and nw width in the ith example                  //
// 4- A[i][0] represents the first channel in the volume, take them from top to down                        //
// input:                                                                                                   //
// Aprev(m,nc_prev,nh_prev,nw_prev) , filters(numOfFilters,nc,f,f) , b(numOfFilters,1)                      //
// output:                                                                                                  //
// A(m,numOfFilters,nh,nw)                                                                                  //
//***********************************************************************************************************/

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::init_LeNet1()
{
    /*ARGUMENT LIST*/
    bool batchNorm = Arg->batchNorm;
    Optimizer optimizer = Arg->optimizer;
    /*END OF ARGUMENT LIST*/

	/*TEMPORARY POINTER*/
	Matrix* Matptr  = nullptr;

    /*FIRST CONVOLUTION LAYER*/
	VectVolume WC1(WC1_NoOfFilters, WC1_Channels, WC1_f, WC1_f, Random_Limited);
	Matrix* bC1 = nullptr;
	if(!batchNorm)
        bC1 =new Matrix(WC1_NoOfFilters, WC1_Channels, 0);

	//Normalizing weights & using Xavier initialization
	float W1mean = 0; float W1std = 0;
	for (int i = 0; i < WC1_NoOfFilters; i++)
		for (int j = 0; j < WC1_Channels; j++)
		{
			W1mean += WC1[i][j]->sumall();
		}
	W1mean /= (WC1_NoOfFilters * WC1_Channels * WC1_f * WC1_f);
	for (int i = 0; i < WC1_NoOfFilters; i++)
	{
		for (int j = 0; j < WC1_Channels; j++)
		{
			//WC1[i][j]=WC1[i][j]-W1mean;
			Matptr = WC1[i][j];
			WC1[i][j] = Matptr->sub(W1mean);
			delete Matptr;
			Matptr = nullptr;

			//W1std+=(WC1[i][j].square()).sumall();
			Matptr = WC1[i][j]->SQUARE();
			W1std = W1std + Matptr->sumall();
			delete Matptr;
			Matptr = nullptr;
		}
	}
	W1std = sqrt(W1std / (WC1_NoOfFilters * WC1_Channels * WC1_f * WC1_f));
	float XavierValue1 = sqrt(2.0 / (WC1_NoOfFilters * WC1_Channels * WC1_f * WC1_f)) / W1std;
	for (int i = 0; i < WC1_NoOfFilters; i++)
		for (int j = 0; j < WC1_Channels; j++)
		{
			//Xavier initialization for relu (*sqrt(2/fin)),try using fout or fin+fout instead of fin...For tanh use 1 instead of 2
			//WC1[i][j] = WC1[i][j] * XavierValue1;
			Matptr = WC1[i][j];
			WC1[i][j] = Matptr->mul(XavierValue1);
			delete Matptr;
			Matptr = nullptr;
		}

	Conv_Weights.put("WC1", WC1);
	if(!batchNorm)
        Conv_biases.put("bC1", bC1);
	/*END OF FIRST CONVOLUTION LAYER*/

	/*SECOND CONVOLUTION LAYER*/
	VectVolume WC2(WC2_NoOfFilters, WC2_Channels, WC2_f, WC2_f, Random_Limited);
	Matrix* bC2 = nullptr;
	if(!batchNorm)
        bC2 = new Matrix(WC2_NoOfFilters, WC2_Channels, 0);

	float W2mean = 0; float W2std = 0;
	for (int i = 0; i < WC2_NoOfFilters; i++)
		for (int j = 0; j < WC2_Channels; j++)
		{
			W2mean += WC2[i][j]->sumall();
		}
	W2mean /= (WC2_NoOfFilters * WC2_Channels * WC2_f * WC2_f);
	for (int i = 0; i < WC2_NoOfFilters; i++)
	{
		for (int j = 0; j < WC2_Channels; j++)
		{
			//WC2[i][j]=WC2[i][j]-W2mean;
			Matptr = WC2[i][j];
			WC2[i][j] = Matptr->sub(W2mean);
			delete Matptr;
			Matptr = nullptr;

			//W2std+=(WC2[i][j].square()).sumall();
			Matptr = WC2[i][j]->SQUARE();
			W2std = W2std + Matptr->sumall();
			delete Matptr;
			Matptr = nullptr;
		}
	}
	W2std = sqrt(W2std / (WC2_NoOfFilters * WC2_Channels * WC2_f * WC2_f));
	float XavierValue2 = sqrt(2.0 / (WC2_NoOfFilters * WC2_Channels * WC2_f * WC2_f)) / W2std;
	for (int i = 0; i < WC2_NoOfFilters; i++)
		for (int j = 0; j < WC2_Channels; j++)
		{
			//Xavier initialization for relu (*sqrt(2/fin)),try using fout or fin+fout instead of fin...For tanh use 1 instead of 2
			//WC2[i][j] = WC2[i][j] * XavierValue2;
			Matptr = WC2[i][j];
			WC2[i][j] = Matptr->mul(XavierValue2);
			delete Matptr;
			Matptr = nullptr;
		}

	Conv_Weights.put("WC2", WC2);
	if(!batchNorm)
        Conv_biases.put("bC2", bC2);
	/*END OF SECOND CONVOLUTION LAYER*/

    /*FULLY CONNECTED LAYERS*/
    int numOfLayers = 2;
    Arg->numOfLayers = numOfLayers;
    delete Arg->layers;
    layer* layers=new layer[numOfLayers];
    layers[0].put(192, NONE);
    layers[1].put(10, SOFTMAX);
    Arg->layers = layers;

    Matrix* W1 = new  Matrix(10,192,Random_Limited);
    Matrix* b1 =nullptr;
    if(!batchNorm)
        b1 =new Matrix(10,1,0);

    //Normalizing weights & using Xavier initialization
    float Wmean = W1->sumall() / (W1->Rows() * W1->Columns());

    //W1 = W1 - Wmean;
    Matptr = W1;
    W1 = Matptr->sub(Wmean);
    delete Matptr;
    Matptr = nullptr;

    //Wstd = sqrt(((W1.square()).sumall()) / (W1.Rows() * W1.Columns()));
    Matptr = W1->SQUARE();
    float Wstd = sqrt((Matptr->sumall()) / (W1->Rows() * W1->Columns()));
    delete Matptr;
    Matptr = nullptr;


    //W1 = W1 / Wstd;
    Matptr = W1;
    W1 = Matptr->div(Wstd);
    delete Matptr;
    Matptr = nullptr;

    //W1 = W1  * sqrt(2.0/layers[0].neurons);
    Matptr = W1;
    W1 = Matptr->mul(sqrt(2.0 / layers[0].neurons));
    delete Matptr;
    Matptr = nullptr;

    //Xavier initialization for relu (*sqrt(2/fin)),try using fout or fin+fout instead of fin...For tanh use 1 instead of 2
    FC_Parameters.put("W1",W1);
    if(!batchNorm)
        FC_Parameters.put("b1",b1);
    /*END OF FULLY CONNECTED LAYERS*/


    /*BATCH NORM INITIALIZATION*/
	if (batchNorm)
	{
		/*Conv layers*/
		//first layer
		Matrix* C1g1 = new Matrix(WC1_NoOfFilters,1,1);
		Matrix* C1g2 = new Matrix(WC1_NoOfFilters, 1);
		Conv_biases.put(CharGen("gC1", 1), C1g1);
		Conv_biases.put(CharGen("gC2", 1), C1g2);
		//second layer
		Matrix* C2g1 = new Matrix(WC2_NoOfFilters, 1, 1);
		Matrix* C2g2 = new Matrix(WC2_NoOfFilters, 1);
		Conv_biases.put(CharGen("gC1", 2), C2g1);
		Conv_biases.put(CharGen("gC2", 2), C2g2);
		/*FC layers*/
		Matrix** g1 = new Matrix*[numOfLayers - 1];   //gamma1
		Matrix** g2 = new Matrix*[numOfLayers - 1];   //gamma2
		for (int ii = 0; ii < numOfLayers - 1; ii++)
		{
			g1[ii] = new Matrix(layers[ii + 1].neurons, 1, 1);
			FC_Parameters.put(CharGen("g1", ii + 1), g1[ii]);
			g2[ii] = new Matrix(layers[ii + 1].neurons, 1);
			FC_Parameters.put(CharGen("g2", ii + 1), g2[ii]);
		}

		/*Intialization For batch Norm at test time*/

		/*Conv layers*/
		//first layer
		Matrix* C1rm= new Matrix(WC1_NoOfFilters, 1);
		Matrix* C1rv = new Matrix(WC1_NoOfFilters, 1);;
		Conv_biases.put("rmC1", C1rm);
		Conv_biases.put("rvC1", C1rv);
		//second layer
		Matrix* C2rm = new Matrix(WC2_NoOfFilters, 1);
		Matrix* C2rv = new Matrix(WC2_NoOfFilters, 1);;
		Conv_biases.put("rmC2", C2rm);
		Conv_biases.put("rvC2", C2rv);
		/*FC layers*/
        Matrix** running_mean = new Matrix*[numOfLayers - 1];     //mean of z for each layer
        Matrix** running_var = new Matrix*[numOfLayers - 1];       //standard deviation of z for each layer
        for (int ii = 0; ii<numOfLayers - 1; ii++)
        {
            running_mean[ii] = new Matrix(layers[ii + 1].neurons, 1);
            FC_Parameters.put(CharGen("rm", ii + 1), running_mean[ii]);
            running_var[ii] = new Matrix(layers[ii + 1].neurons, 1);
            FC_Parameters.put(CharGen("rv", ii + 1), running_var[ii]);
        }
    }
	/*END OF BATCHNORM INITIALIZATION*/

	/*ADAM INITIALIZATION*/
	if (optimizer == ADAM)
	{
	    /* ConvLayer 1 */
		VectVolume SdwC1(WC1_NoOfFilters, WC1_Channels, WC1_f, WC1_f);
		ADAM_dWC.put("SdwC1", SdwC1);
		VectVolume VdwC1(WC1_NoOfFilters, WC1_Channels, WC1_f, WC1_f);
		ADAM_dWC.put("VdwC1", VdwC1);
		if(!batchNorm)
        {
            Matrix* SdbC1 = new Matrix(WC1_NoOfFilters, WC1_Channels);
            ADAM_dbC.put("SdbC1", SdbC1);
            Matrix* VdbC1 = new Matrix(WC1_NoOfFilters, WC1_Channels);
            ADAM_dbC.put("VdbC1", VdbC1);
        }

		/* ConvLayer 2 */
		VectVolume SdwC2(WC2_NoOfFilters, WC2_Channels, WC2_f, WC2_f);
		ADAM_dWC.put("SdwC2", SdwC2);
		VectVolume VdwC2(WC2_NoOfFilters, WC2_Channels, WC2_f, WC2_f);
		ADAM_dWC.put("VdwC2", VdwC2);
        if(!batchNorm)
        {
            Matrix* SdbC2 = new Matrix(WC2_NoOfFilters, WC2_Channels);
            ADAM_dbC.put("SdbC2", SdbC2);
            Matrix* VdbC2 = new Matrix(WC2_NoOfFilters, WC2_Channels);
            ADAM_dbC.put("VdbC2", VdbC2);
        }
		else
		{
			//first layer
			Matrix* C1sdg1 = new Matrix(WC1_NoOfFilters, 1);
			Matrix* C1vdg1 = new Matrix(WC1_NoOfFilters, 1);
			Matrix* C1sdg2 = new Matrix(WC1_NoOfFilters, 1);
			Matrix* C1vdg2 = new Matrix(WC1_NoOfFilters, 1);
			ADAM_dbC.put("sdg1C1", C1sdg1);
			ADAM_dbC.put("vdg1C1", C1vdg1);
			ADAM_dbC.put("sdg2C1", C1sdg2);
			ADAM_dbC.put("vdg2C1", C1vdg2);
			//second layer
			Matrix* C2sdg1 = new Matrix(WC2_NoOfFilters, 1);
			Matrix* C2vdg1 = new Matrix(WC2_NoOfFilters, 1);
			Matrix* C2sdg2 = new Matrix(WC2_NoOfFilters, 1);
			Matrix* C2vdg2 = new Matrix(WC2_NoOfFilters, 1);
			ADAM_dbC.put("sdg1C2", C2sdg1);
			ADAM_dbC.put("vdg1C2", C2vdg1);
			ADAM_dbC.put("sdg2C2", C2sdg2);
			ADAM_dbC.put("vdg2C2", C2vdg2);
		}

	    /* FC_Layer */
		Matrix* Sdw1 = new Matrix(10, 192);
		FC_ADAM.put("Sdw1", Sdw1);
		Matrix* Vdw1 = new Matrix(10, 192);
		FC_ADAM.put("Vdw1", Vdw1);
		if(!batchNorm)
        {
            Matrix* Sdb1 = new Matrix(10, 1);
            FC_ADAM.put("Sdb1", Sdb1);
            Matrix* Vdb1 = new Matrix(10, 1);
            FC_ADAM.put("Vdb1", Vdb1);
        }

	    Matrix** sdg1 = new Matrix*[numOfLayers - 1];
        Matrix** vdg1 = new Matrix*[numOfLayers - 1];
        Matrix** sdg2 = new Matrix*[numOfLayers - 1];
        Matrix** vdg2 = new Matrix*[numOfLayers - 1];
		for (int i = 0; i < numOfLayers - 1; i++)   // L-1 = number of hidden layers + output layer
		{
			if (batchNorm)
			{
				sdg1[i] = new Matrix(layers[i + 1].neurons, 1);
				FC_ADAM.put(CharGen("sg1", i + 1), sdg1[i]);
				vdg1[i] = new Matrix(layers[i + 1].neurons, 1);
				FC_ADAM.put(CharGen("vg1", i + 1), vdg1[i]);

				sdg2[i] = new Matrix(layers[i + 1].neurons, 1);
				FC_ADAM.put(CharGen("sg2", i + 1), sdg2[i]);
				vdg2[i] = new Matrix(layers[i + 1].neurons, 1);
				FC_ADAM.put(CharGen("vg2", i + 1), vdg2[i]);
			}
		}
	}
	/*END OF ADAM INITIALIZATION*/
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::train_LeNet1()
{
    /*ARGUMENT LIST*/
    Matrix* X = Arg->X;
    Matrix* Y = Arg->Y;
    int minibatchSize = Arg->batchSize;
    int numOfEpochs = Arg->numOfEpochs;
    int numOfLayers = Arg->numOfLayers;
    layer* layers = Arg->layers;
    bool dropout = Arg->dropout;
    bool dropConnect = Arg->dropConnect;
    float* keep_prob = Arg->keep_prob;
    /*END OF ARGUMENT LIST*/

	/*ESSENTIAL VARIABLE INITIALIZATION*/
    isLastepoch = false;
    momentum = true;
    int t = 0;              //Counter used for Adam optimizer
    /*END OF ESSENTIAL VARIABLE INITIALIZATION*/

	/*BEGINNING OF EPOCHS ITERATIONS*/
	for (int i = 0; i < numOfEpochs; i++)
	{
        clock_t start = clock();

        /*Iterations on mini batches*/
		int m = X->Columns();
		int numOfMiniBatches = m / minibatchSize;
		int LastBatchSize = m - minibatchSize * numOfMiniBatches;
		int j;
		if (i == numOfEpochs - 1)
			isLastepoch = true;

        for (j = 0; j < numOfMiniBatches; j++)
		{
		    if(dropout)
            {
                for (int k = 0; k < numOfLayers - 1; k++)
                    D[Cores][k + 1] = new Matrix(layers[k + 1].neurons, minibatchSize, Bernoulli, keep_prob[k + 1]);
            }
            if(dropConnect)
            {
                for(int jj = 0; jj < numOfLayers - 2; jj++)
                {
                    D2[Cores][jj] = new matrix<bool>*[minibatchSize];
                    for(int kk = 0; kk < minibatchSize; kk++)
                    {
                        D2[Cores][jj][kk] = new matrix<bool>(layers[jj + 1].neurons, layers[jj].neurons, Bernoulli, keep_prob[jj + 1]);
                    }
                }
            }

		    Matrix* cur_X = X->SubMat(0, j*minibatchSize, X->Rows() - 1, ((j + 1)*(minibatchSize)-1));
            Matrix* cur_Y = Y->SubMat(0, j*minibatchSize, Y->Rows() - 1, ((j + 1)*(minibatchSize)-1));
            int m = cur_X->Columns();

            VectVolume AC0 = to_VectorOfVolume(cur_X, 28, 28, 1, m);
            Conv_Cache[Cores].put("AC0", AC0);
            convLayer(TRAIN,1, 1, LEAKYRELU, Cores);              //stride=1, A_index=1,W_index=1
            poolLayer(2, 2, AVG, 1, Cores);                 //stride=2, f=2 ,mode ="avg",A_index=1
            convLayer(TRAIN, 1, 2, LEAKYRELU, Cores);              //stride=1 ,A_index=2,W_index=2
            poolLayer(2, 2, AVG, 2, Cores);                 //f=5 ,mode ="avg",A_index=2
            VectVolume ACP2 = Conv_Cache[Cores]["ACP2"];
            Matrix* A0 = to_FC(ACP2);
            FC_Cache[Cores].put("A0", A0);
            Matrix* Y_hat = FC_FeedForward(TRAIN, Cores);
            ///////////////////////////////////////////////////////
            FC_CalGrads(cur_Y, Y_hat, Cores);
            FC_UpdateParameters(t, Cores);
            Matrix* dA0 = FC_Grades[Cores]["dA0"];
            VectVolume dACP2 = to_VectorOfVolume(dA0, ACP2[0][0]->Rows(), ACP2[0][0]->Columns(), ACP2[0].size(), m);
            Conv_Grades[Cores].put("dACP2", dACP2);
            pool_backward(2, 2, AVG, 2, Cores);			  //f=2,stride=2,mode ="avg",A_index=2
            ConvBackward(1, 2, LEAKYRELU, Cores);   //stride=1 ,A_index=2,W_index=2
            pool_backward(2, 2, AVG, 1, Cores);			  //f=2,stride=2,mode ="avg",A_index=1
            ConvBackward(1, 1, LEAKYRELU, Cores);   //stride=1 ,A_index=1,W_index=1
            Conv_updateparameters(t, 2, Cores);
            Conv_updateparameters(t, 1, Cores);
            t++;

            delete cur_X;
            delete cur_Y;
            FC_Cache[Cores].DeleteThenClear();
            Conv_Cache[Cores].DeleteThenClearObj();
			Conv_Cache_Mat[Cores].DeleteThenClear();
            FC_Grades[Cores].DeleteThenClear();
            Conv_dbiases[Cores].DeleteThenClear();
            Conv_Grades[Cores].DeleteThenClearObj();
            cout << endl << "Minibatch No." << j << " ended , Time = " << double(clock() - start) / CLOCKS_PER_SEC << endl;
		}

		if (LastBatchSize != 0)
		{
			Matrix* cur_X = X->SubMat(0, j*minibatchSize, X->Rows() - 1, X->Columns() - 1);
			Matrix* cur_Y = Y->SubMat(0, j*minibatchSize, Y->Rows() - 1, Y->Columns() - 1);
			int m = cur_X->Columns();
			VectVolume AC0 = to_VectorOfVolume(cur_X, 28, 28, 1, m);
			Conv_Cache[Cores].put("AC0", AC0);
			convLayer(TRAIN, 1, 1, LEAKYRELU, Cores);             //stride=1 ,A_index=1,W_index=1
			poolLayer(2, 2, AVG, 1, Cores);           //f=5 ,mode ="avg",A_index=1
			convLayer(TRAIN, 1, 2, LEAKYRELU, Cores);             //stride=1 ,A_index=3,W_index=2
			poolLayer(2, 2, AVG, 2, Cores);           //f=5 ,mode ="average",A_index=2
			VectVolume ACP2 = Conv_Cache[Cores]["ACP2"];
			Matrix* A0 = to_FC(ACP2);
			FC_Cache[Cores].put("A0", A0);
			Matrix* Y_hat = FC_FeedForward(TRAIN, Cores);
			///////////////////////////////////////////////////////
			FC_CalGrads(cur_Y, Y_hat, Cores);
			Matrix* dA0 = FC_Grades[Cores]["dA0"];
			VectVolume dACP2 = to_VectorOfVolume(dA0, ACP2[0][0]->Rows(), ACP2[0][0]->Columns(), ACP2[0].size(), m);
			Conv_Grades[Cores].put("dACP2", dACP2);
			pool_backward(2, 2, AVG, 2, Cores);               //f=2,stride=2,mode ="avg",A_index=2
			ConvBackward(1, 2, LEAKYRELU, Cores);         //stride=1 ,A_index=2,W_index=2
			pool_backward(2, 2, AVG, 1, Cores);			   //f=2,stride=2,mode ="avg",A_index=1
			ConvBackward(1, 1, LEAKYRELU, Cores);         //stride=1 ,A_index=1,W_index=1

			FC_UpdateParameters(t, Cores);
			Conv_updateparameters(t, 2, Cores);
			Conv_updateparameters(t, 1, Cores);
			t++;

			delete cur_X;
			delete cur_Y;
			FC_Cache[Cores].DeleteThenClear();
			Conv_Cache[Cores].DeleteThenClearObj();
			Conv_Cache_Mat[Cores].DeleteThenClear();
			FC_Grades[Cores].DeleteThenClear();
			Conv_dbiases[Cores].DeleteThenClear();
			Conv_Grades[Cores].DeleteThenClearObj();   //Because updateparameters is called twice so we can't put it inside ConvBackward
		}
    }
	/*END OF EPOCHS ITERATIONS*/
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::train_LeNet1_thread()
{
    /*ARGUMENT LIST*/
    Matrix* X = Arg->X;
    Matrix* Y = Arg->Y;
    int minibatchSize = Arg->batchSize;
    int numOfEpochs = Arg->numOfEpochs;
    int numOfLayers = Arg->numOfLayers;
    layer* layers = Arg->layers;
    float* keep_prob = Arg->keep_prob;
    bool dropout = Arg->dropout;
    /*END OF ARGUMENT LIST*/

    /*ESSENTIAL VARIABLE INITIALIZATION*/
    isLastepoch = false;
    momentum = true;
    int t = 0;              //Counter used for Adam optimizer
    /*END OF ESSENTIAL VARIABLE INITIALIZATION*/

    /*BEGINNING OF EPOCHS ITERATIONS*/
	for (int i = 0; i < numOfEpochs; i++)
	{
		clock_t start = clock();
		/*Iterations on mini batches*/
		int m = X->Columns();
		int numOfMiniBatches = m / minibatchSize;
		int LastBatchSize = m - minibatchSize * numOfMiniBatches;
		int j;
		if (i == numOfEpochs - 1)
			isLastepoch = true;

		for (j = 0; j < numOfMiniBatches; j++)
		{
		    //Drop-out preparation
			if(dropout)
            {
                for (int k = 0; k < numOfLayers - 1; k++)
                    D[0][k + 1] = new Matrix(layers[k + 1].neurons, minibatchSize / Cores, Bernoulli, keep_prob[k + 1]);
                for (int ii = 1; ii < Cores; ii++)
                    for (int jj = 0; jj < numOfLayers - 1; jj++)
                    {
                        D[ii][jj + 1] =  new Matrix(layers[jj + 1].neurons, minibatchSize / Cores);
                        *(D[ii][jj + 1]) = *(D[0][jj + 1]);
                    }
            }

			thread** Threads = new thread*[Cores];
			for (int k = 0; k < Cores; k++)
			{
				int start = (j*minibatchSize) + ((minibatchSize * k) / Cores);
				int end = (j*minibatchSize) + (minibatchSize * (k + 1) / Cores) - 1;

				Threads[k] = new thread(Thread_Initialize(), this, k, start, end);
			}

			for (int k = 0; k < Cores; k++)
			{
				Threads[k]->join();
				delete Threads[k];
			}
            delete Threads;

			//Average the gradients of the differnet caches and put them in FC_Grades[Cores]
			Average();

			FC_UpdateParameters(t, Cores);
			Conv_updateparameters(t, 2, Cores);
			Conv_updateparameters(t, 1, Cores);
			t++;

			for (int k = 0; k < Cores + 1; k++)
			{
				FC_Grades[k].DeleteThenClear();
				Conv_dbiases[k].DeleteThenClear();
				Conv_Grades[k].DeleteThenEraseObj("dWC1");
				Conv_Grades[k].DeleteThenEraseObj("dWC2");
				//Because updateparameters is called twice so we can't put it inside ConvBackward
			}

			cout << endl << "Minibatch No." << j << " ended , Time = " << double(clock() - start) / CLOCKS_PER_SEC <<endl;
		}

		if (LastBatchSize != 0)
		{
			Matrix* cur_X = X->SubMat(0, j*minibatchSize, X->Rows() - 1, X->Columns() - 1);
			Matrix* cur_Y = Y->SubMat(0, j*minibatchSize, Y->Rows() - 1, Y->Columns() - 1);
			int m = cur_X->Columns();
			VectVolume AC0 = to_VectorOfVolume(cur_X, 28, 28, 1, m);
			Conv_Cache[Cores].put("AC0", AC0);
			convLayer(TRAIN, 1, 1, LEAKYRELU, Cores);             //stride=1 ,A_index=1,W_index=1
			poolLayer(2, 2, AVG, 1, Cores);           //f=5 ,mode ="avg",A_index=1
			convLayer(TRAIN, 1, 2, LEAKYRELU, Cores);             //stride=1 ,A_index=3,W_index=2
			poolLayer(2, 2, AVG, 2, Cores);           //f=5 ,mode ="average",A_index=2
			VectVolume ACP2 = Conv_Cache[Cores]["ACP2"];
			Matrix* A0 = to_FC(ACP2);
			FC_Cache[Cores].put("A0", A0);
			Matrix* Y_hat = FC_FeedForward(TRAIN, Cores);
			///////////////////////////////////////////////////////
			FC_CalGrads(cur_Y, Y_hat, Cores);
			Matrix* dA0 = FC_Grades[Cores]["dA0"];
			VectVolume dACP2 = to_VectorOfVolume(dA0, ACP2[0][0]->Rows(), ACP2[0][0]->Columns(), ACP2[0].size(), m);

			Conv_Grades[Cores].put("dACP2", dACP2);
			pool_backward(2, 2, AVG, 2, Cores);               //f=2,stride=2,mode ="avg",A_index=2
			ConvBackwardOptimized(1, 2, LEAKYRELU, Cores);         //stride=1 ,A_index=2,W_index=2
			pool_backward(2, 2, AVG, 1, Cores);			   //f=2,stride=2,mode ="avg",A_index=1
			ConvBackwardOptimized(1, 1, LEAKYRELU, Cores);         //stride=1 ,A_index=1,W_index=1

			FC_UpdateParameters(t, Cores);
			Conv_updateparameters(t, 2, Cores);
			Conv_updateparameters(t, 1, Cores);
			t++;

			delete cur_X;
			delete cur_Y;
			FC_Cache[Cores].DeleteThenClear();
			Conv_Cache[Cores].DeleteThenClearObj();
			FC_Grades[Cores].DeleteThenClear();
			Conv_dbiases[Cores].DeleteThenClear();
			Conv_Grades[Cores].DeleteThenClearObj();   //Because updateparameters is called twice so we can't put it inside ConvBackward
		}
	}
	/*END OF EPOCHS ITERATIONS*/
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* NeuralNetwork::test_LeNet1(Mode mode) //forward m examples and return the predictions in Y_hat
{
    /*ARGUMENT LIST*/
    Matrix* X_dev = Arg->X_dev;
    Matrix* X_test = Arg->X_test;
    /*END OF ARGUMENT LIST*/

	Matrix * Y_hat = nullptr;
	VectVolume AC0;

	if(mode == DEV)
    {
        int m = X_dev->Columns();
        AC0 = to_VectorOfVolume(X_dev,28,28,1,m);
    }
    else
    {
        int m = X_test->Columns();
        AC0 = to_VectorOfVolume(X_test,28,28,1,m);
    }
    Conv_Cache[Cores].put("AC0",AC0);
    convLayer(TEST,1,1,LEAKYRELU,Cores);           //stride=1 ,A_index=1,W_index=1
    poolLayer(2,2, AVG,1,Cores);                    //f=5 ,mode ="avg",A_index=1
    convLayer(TEST, 1,2,LEAKYRELU,Cores);           //stride=1 ,A_index=3,W_index=2
    poolLayer(2,2, AVG,2,Cores);         //f=5 ,mode ="avg",A_index=2
	VectVolume ACP2 = Conv_Cache[Cores]["ACP2"];
    Matrix* A0 = to_FC(ACP2);
	Conv_Cache[Cores].DeleteThenClearObj();
	Conv_Cache_Mat[Cores].DeleteThenClear();
    FC_Cache[Cores].put("A0",A0);
    Y_hat = FC_FeedForward(TEST, Cores);
	return Y_hat;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
