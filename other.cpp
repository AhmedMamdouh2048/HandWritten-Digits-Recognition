#include "NeuralNetwork.h"
# define AT LEAKYRELU
#define WC1_NoOfFilters 4
#define WC1_Channels 1
#define WC1_f 5
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::init_other()
{
     /*ARGUMENT LIST*/
    bool batchNorm = Arg->batchNorm;
    Optimizer optimizer = Arg->optimizer;
    /*END OF ARGUMENT LIST*/

	/*Temporal variables*/
	Matrix*  MatPtr = nullptr;

	float Xnum;
	if (AT == SATLINEAR || AT == SATLINEAR2 || AT == SATLINEAR3 || AT == TANH)
		Xnum = 1.0;
	else
		Xnum = 2.0;
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
			MatPtr = WC1[i][j];
			WC1[i][j] = MatPtr->sub(W1mean);
			delete MatPtr;
			MatPtr = nullptr;

			//W1std+=(WC1[i][j].square()).sumall();
			MatPtr = WC1[i][j]->SQUARE();
			W1std = W1std + MatPtr->sumall();
			delete MatPtr;
			MatPtr = nullptr;
		}
	}
	W1std = sqrt(W1std / (WC1_NoOfFilters * WC1_Channels * WC1_f * WC1_f));
	float XavierValue1 = sqrt(Xnum / (WC1_NoOfFilters * WC1_Channels * WC1_f * WC1_f)) / W1std;
	for (int i = 0; i < WC1_NoOfFilters; i++)
		for (int j = 0; j < WC1_Channels; j++)
		{
			//Xavier initialization for relu (*sqrt(2/fin)),try using fout or fin+fout instead of fin...For tanh use 1 instead of 2
			//WC1[i][j] = WC1[i][j] * XavierValue1;
			MatPtr = WC1[i][j];
			WC1[i][j] = MatPtr->mul(XavierValue1);
			delete MatPtr;
			MatPtr = nullptr;
		}

	Conv_Weights.put("WC1", WC1);
	if(!batchNorm)
        Conv_biases.put("bC1", bC1);
    /*END OF FIRST CONVOLUTION LAYER*/

	/*Fully Connected Layers*/
	int numOfLayers = 4;
    Arg->numOfLayers = numOfLayers;
    delete Arg->layers;
    layer* layers=new layer[numOfLayers];
	layers[0].put(576, NONE);
	layers[1].put(400, LEAKYRELU);
	layers[2].put(200, LEAKYRELU);
	layers[3].put(10, SOFTMAX);
    Arg->layers = layers;

	Matrix** Mw = new Matrix*[numOfLayers - 1];
	Matrix** Mb = new Matrix*[numOfLayers - 1];

	for (int i = 0; i<numOfLayers - 1; i++)  // L-1 = number of hidden layers + output layer
	{
		Mw[i] = new Matrix(layers[i + 1].neurons, layers[i].neurons, Random); // Mw[0] holds W1 and so on
		MatPtr = Mw[i];
		Mw[i] = Mw[i]->div(float(RAND_MAX));
		delete MatPtr;

		/*To make the standard deviation of weights = 1 and mean = 0*/
		if (Mw[i]->Rows() != 1 || Mw[i]->Columns() != 1) // Don't calculate if dimensions are 1x1
		{
			float Wmean = Mw[i]->sumall() / (Mw[i]->Rows() * Mw[i]->Columns());
			MatPtr = Mw[i];
			Mw[i] = Mw[i]->sub(Wmean);
			delete MatPtr;

			float Wstd = sqrt((Mw[i]->square()).sumall() / (Mw[i]->Rows() * Mw[i]->Columns()));
			MatPtr = Mw[i];
			Mw[i] = Mw[i]->div(Wstd);
			delete MatPtr;
		}

		if (layers[i + 1].activation == SIGMOID)
		{
			MatPtr = Mw[i];
			Mw[i] = Mw[i]->mul(sqrt(2 / layers[i].neurons));
			delete MatPtr;
		}
		else if (layers[i + 1].activation == SOFTMAX)
		{
			MatPtr = Mw[i];
			Mw[i] = Mw[i]->mul(sqrt(2 / layers[i].neurons));
			delete MatPtr;
		}
		else if (layers[i + 1].activation == TANH)
		{
			MatPtr = Mw[i];
			Mw[i] = Mw[i]->mul(sqrt(1 / layers[i].neurons));
			delete MatPtr;
		}
		else if (layers[i + 1].activation == RELU)
		{
			MatPtr = Mw[i];
			Mw[i] = Mw[i]->mul(sqrt(2 / layers[i].neurons));
			delete MatPtr;
		}
		else if (layers[i + 1].activation == LEAKYRELU)
		{
			MatPtr = Mw[i];
			Mw[i] = Mw[i]->mul(sqrt(2 / layers[i].neurons));
			delete MatPtr;
		}
		else if (layers[i + 1].activation == SATLINEAR)
		{
			MatPtr = Mw[i];
			Mw[i] = Mw[i]->mul(sqrt(1 / layers[i].neurons));
			delete MatPtr;
		}
		else if (layers[i + 1].activation == LINEAR)
		{
			MatPtr = Mw[i];
			Mw[i] = Mw[i]->mul(sqrt(2 / layers[i].neurons));
			delete MatPtr;
		}
		else if (layers[i + 1].activation == SATLINEAR2)
		{
			MatPtr = Mw[i];
			Mw[i] = Mw[i]->mul(sqrt(1 / layers[i].neurons));
			delete MatPtr;
		}
		else if (layers[i + 1].activation == SATLINEAR3)
		{
			MatPtr = Mw[i];
			Mw[i] = Mw[i]->mul(sqrt(1 / layers[i].neurons));
			delete MatPtr;
		}
		FC_Parameters.put(CharGen("W", i + 1), Mw[i]);

		if (!batchNorm)
		{
			Mb[i] = new Matrix(layers[i + 1].neurons, 1, Random);
			MatPtr = Mb[i];
			Mb[i] = Mb[i]->div(float(RAND_MAX));
			delete MatPtr;

			/*To make the standard deviation of biases = 1 and mean = 0*/
			if (Mb[i]->Rows() != 1 || Mb[i]->Columns() != 1)         // Don't calculate if dimensions are 1x1
			{
				float bmean = Mb[i]->sumall() / (Mb[i]->Rows() * Mb[i]->Columns());
				MatPtr = Mb[i];
				Mb[i] = Mb[i]->sub(bmean);
				delete MatPtr;

				float bstd = sqrt((Mb[i]->square()).sumall() / (Mb[i]->Rows() * Mb[i]->Columns()));
				MatPtr = Mb[i];
				Mb[i] = Mb[i]->div(bstd);
				delete MatPtr;
			}

			if (layers[i + 1].activation == SIGMOID)
			{
				MatPtr = Mb[i];
				Mb[i] = Mb[i]->mul(sqrt(2 / layers[i].neurons));
				delete MatPtr;
			}
			else if (layers[i + 1].activation == SOFTMAX)
			{
				MatPtr = Mb[i];
				Mb[i] = Mb[i]->mul(sqrt(2 / layers[i].neurons));
				delete MatPtr;
			}
			else if (layers[i + 1].activation == TANH)
			{
				MatPtr = Mb[i];
				Mb[i] = Mb[i]->mul(sqrt(1 / layers[i].neurons));
				delete MatPtr;
			}
			else if (layers[i + 1].activation == RELU)
			{
				MatPtr = Mb[i];
				Mb[i] = Mb[i]->mul(sqrt(2 / layers[i].neurons));
				delete MatPtr;
			}
			else if (layers[i + 1].activation == LEAKYRELU)
			{
				MatPtr = Mb[i];
				Mb[i] = Mb[i]->mul(sqrt(2 / layers[i].neurons));
				delete MatPtr;
			}
			else if (layers[i + 1].activation == SATLINEAR)
			{
				MatPtr = Mb[i];
				Mb[i] = Mb[i]->mul(1 / layers[i].neurons);
				delete MatPtr;
			}
			else if (layers[i + 1].activation == LINEAR)
			{
				MatPtr = Mb[i];
				Mb[i] = Mb[i]->mul(sqrt(2 / layers[i].neurons));
				delete MatPtr;
			}
			else if (layers[i + 1].activation == SATLINEAR2)
			{
				MatPtr = Mb[i];
				Mb[i] = Mb[i]->mul(1 / layers[i].neurons);
				delete MatPtr;
			}
			else if (layers[i + 1].activation == SATLINEAR3)
			{
				MatPtr = Mb[i];
				Mb[i] = Mb[i]->mul(1 / layers[i].neurons);
				delete MatPtr;
			}
			FC_Parameters.put(CharGen("b", i + 1), Mb[i]);
		}
	}
   /*BATCH NORM INITIALIZATION*/
	if (batchNorm)
	{
		/*Conv layers*/
		//first layer
		Matrix* C1g1 = new Matrix(WC1_NoOfFilters,1,1);
		Matrix* C1g2 = new Matrix(WC1_NoOfFilters, 1);
		Conv_biases.put(CharGen("gC1", 1), C1g1);
		Conv_biases.put(CharGen("gC2", 1), C1g2);

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
		}

	    /* FC_Layer */
		Matrix** Msdw = new Matrix*[numOfLayers - 1];
        Matrix** Mvdw = new Matrix*[numOfLayers - 1];
        Matrix** Msdb = new Matrix*[numOfLayers - 1];
        Matrix** Mvdb = new Matrix*[numOfLayers - 1];

        //For batchnorm
        Matrix** sdg1 = new Matrix*[numOfLayers - 1];
        Matrix** vdg1 = new Matrix*[numOfLayers - 1];
        Matrix** sdg2 = new Matrix*[numOfLayers - 1];
        Matrix** vdg2 = new Matrix*[numOfLayers - 1];

		for (int i = 0; i < numOfLayers - 1; i++)   // L-1 = number of hidden layers + output layer
		{
			Msdw[i] = new Matrix(layers[i + 1].neurons, layers[i].neurons);
			FC_ADAM.put(CharGen("Sdw", i + 1), Msdw[i]);

			Mvdw[i] = new Matrix(layers[i + 1].neurons, layers[i].neurons);
			FC_ADAM.put(CharGen("Vdw", i + 1), Mvdw[i]);
            if(!batchNorm)
            {
                Msdb[i] = new Matrix(layers[i + 1].neurons, 1);
                FC_ADAM.put(CharGen("Sdb", i + 1), Msdb[i]);

                Mvdb[i] = new Matrix(layers[i + 1].neurons, 1);
                FC_ADAM.put(CharGen("Vdb", i + 1), Mvdb[i]);
            }
			else
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
void NeuralNetwork::train_other()
{
    /*ARGUMENT LIST*/
    IntMatrix* X = Arg->X;
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

		    Matrix* cur_X = X->Sub_Mat(0, j*minibatchSize, X->Rows() - 1, ((j + 1)*(minibatchSize)-1));
            Matrix* cur_Y = Y->Sub_Mat(0, j*minibatchSize, Y->Rows() - 1, ((j + 1)*(minibatchSize)-1));
            int m = cur_X->Columns();

            VectVolume AC0 = to_VectorOfVolume(cur_X, 28, 28, 1, m);
            Conv_Cache[Cores].put("AC0", AC0);
            convLayer(TRAIN,1, 1, AT, Cores);              //stride=1, A_index=1,W_index=1
            poolLayer(2, 2, AVG, 1, Cores);                       //stride=2, f=2 ,mode ="avg",A_index=1
            VectVolume ACP1 = Conv_Cache[Cores]["ACP1"];
            Matrix* A0 = to_FC(ACP1);
            FC_Cache[Cores].put("A0", A0);
            Matrix* Y_hat = FC_FeedForward(TRAIN, Cores);
            ///////////////////////////////////////////////////////
            FC_CalGrads(cur_Y, Y_hat, Cores);
            FC_UpdateParameters(t, Cores);
            Matrix* dA0 = FC_Grades[Cores]["dA0"];
			VectVolume dACP1 = to_VectorOfVolume(dA0, ACP1[0][0]->Rows(), ACP1[0][0]->Columns(), ACP1[0].size(), m);

			Conv_Grades[Cores].put("dACP1", dACP1);
			pool_backward(2, 2, AVG, 1, Cores);			   //f=2,stride=2,mode ="avg",A_index=1
			ConvBackward(1, 1, AT, Cores);                 //stride=1 ,A_index=1,W_index=1
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
			Matrix* cur_X = X->Sub_Mat(0, j*minibatchSize, X->Rows() - 1, X->Columns() - 1);
			Matrix* cur_Y = Y->Sub_Mat(0, j*minibatchSize, Y->Rows() - 1, Y->Columns() - 1);
			int m = cur_X->Columns();
			VectVolume AC0 = to_VectorOfVolume(cur_X, 28, 28, 1, m);
			Conv_Cache[Cores].put("AC0", AC0);
			convLayer(TRAIN, 1, 1, AT, Cores);             //stride=1 ,A_index=1,W_index=1
			poolLayer(2, 2, AVG, 1, Cores);           //f=5 ,mode ="avg",A_index=1
			 VectVolume ACP1 = Conv_Cache[Cores]["ACP1"];
            Matrix* A0 = to_FC(ACP1);
            FC_Cache[Cores].put("A0", A0);
            Matrix* Y_hat = FC_FeedForward(TRAIN, Cores);
            ///////////////////////////////////////////////////////
            FC_CalGrads(cur_Y, Y_hat, Cores);
            FC_UpdateParameters(t, Cores);
            Matrix* dA0 = FC_Grades[Cores]["dA0"];
			VectVolume dACP1 = to_VectorOfVolume(dA0, ACP1[0][0]->Rows(), ACP1[0][0]->Columns(), ACP1[0].size(), m);

			Conv_Grades[Cores].put("dACP1", dACP1);
			pool_backward(2, 2, AVG, 1, Cores);			   //f=2,stride=2,mode ="avg",A_index=1
			ConvBackward(1, 1, AT, Cores);         //stride=1 ,A_index=1,W_index=1
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

		}
	}
    /*END OF EPOCHS ITERATIONS*/
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::train_other_thread()
{
     /*ARGUMENT LIST*/
    IntMatrix* X = Arg->X;
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
			Conv_updateparameters(t, 1, Cores);
			t++;

			for (int k = 0; k < Cores + 1; k++)
			{
				FC_Grades[k].DeleteThenClear();
				Conv_dbiases[k].DeleteThenClear();
				Conv_Grades[k].DeleteThenEraseObj("dWC1");
				//Because updateparameters is called twice so we can't put it inside ConvBackward
			}
			cout << endl << "Minibatch No." << j << " ended , Time = " << double(clock() - start) / CLOCKS_PER_SEC <<endl;
		}

		if (LastBatchSize != 0)
		{
			Matrix* cur_X = X->Sub_Mat(0, j*minibatchSize, X->Rows() - 1, X->Columns() - 1);
			Matrix* cur_Y = Y->Sub_Mat(0, j*minibatchSize, Y->Rows() - 1, Y->Columns() - 1);
			int m = cur_X->Columns();
			VectVolume AC0 = to_VectorOfVolume(cur_X, 28, 28, 1, m);
			Conv_Cache[Cores].put("AC0", AC0);
			convLayer(TRAIN, 1, 1, AT, Cores);             //stride=1 ,A_index=1,W_index=1
			poolLayer(2, 2, AVG, 1, Cores);           //f=5 ,mode ="avg",A_index=1
			 VectVolume ACP1 = Conv_Cache[Cores]["ACP1"];
            Matrix* A0 = to_FC(ACP1);
            FC_Cache[Cores].put("A0", A0);
            Matrix* Y_hat = FC_FeedForward(TRAIN, Cores);
            ///////////////////////////////////////////////////////
            FC_CalGrads(cur_Y, Y_hat, Cores);
            FC_UpdateParameters(t, Cores);
            Matrix* dA0 = FC_Grades[Cores]["dA0"];
			VectVolume dACP1 = to_VectorOfVolume(dA0, ACP1[0][0]->Rows(), ACP1[0][0]->Columns(), ACP1[0].size(), m);

			Conv_Grades[Cores].put("dACP1", dACP1);
			pool_backward(2, 2, AVG, 1, Cores);			   //f=2,stride=2,mode ="avg",A_index=1
			ConvBackward(1, 1, AT, Cores);         //stride=1 ,A_index=1,W_index=1
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
		}
	}
	/*END OF EPOCHS ITERATIONS*/
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* NeuralNetwork::test_other(Mode mode) //forward m examples and return the predictions in Y_hat
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
    convLayer(TEST,1,1,AT,Cores);           //stride=1 ,A_index=1,W_index=1
    poolLayer(2,2, AVG,1,Cores);                    //f=5 ,mode ="avg",A_index=1
	VectVolume ACP1 = Conv_Cache[Cores]["ACP1"];
    Matrix* A0 = to_FC(ACP1);
	Conv_Cache[Cores].DeleteThenClearObj();
	Conv_Cache_Mat[Cores].DeleteThenClear();
    FC_Cache[Cores].put("A0",A0);
    Y_hat = FC_FeedForward(TEST, Cores);
	return Y_hat;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
