#include "NeuralNetwork.h"

#define dWC1_NoOfFilters 4
#define dWC1_Channels 1
#define dWC1_f 5
#define dWC2_NoOfFilters 12
#define dWC2_Channels 4
#define dWC2_f 5

void NeuralNetwork::Thread_Initialize::operator()(NeuralNetwork* NN, int ThreadNum, int start, int end)
{
    /*ARGUMENT LIST*/
    TypeOfNet NetType = NN->Arg->NetType;
    IntMatrix* X = NN->Arg->X;
    Matrix* Y = NN->Arg->Y;
    bool dropConnect = NN->Arg->dropConnect;
    int numOfLayers = NN->Arg->numOfLayers;
    int minibatchSize = NN->Arg->batchSize;
    int Cores = NN->Cores;
    float* keep_prob = NN->Arg->keep_prob;
    layer* layers = NN->Arg->layers;
    /*END OF ARGUMENT LIST*/

	if (NetType == FC)
	{
	    if(dropConnect)
        {
            for(int jj = 0; jj < numOfLayers - 2; jj++)
            {
                NN->D2[ThreadNum][jj] = new matrix<bool>*[minibatchSize / Cores];
                for(int kk = 0; kk < minibatchSize / Cores; kk++)
                {
                    NN->D2[ThreadNum][jj][kk] = new matrix<bool>(layers[jj + 1].neurons, layers[jj].neurons, Bernoulli, keep_prob[jj + 1]);
                }
            }
        }
		Matrix* cur_X = X->Sub_Mat(0, start, X->Rows() - 1, end);
		Matrix* cur_Y = Y->Sub_Mat(0, start, Y->Rows() - 1, end);
		NN->FC_Cache[ThreadNum].put("A0", cur_X);
		Matrix* Y_hat = NN->FC_FeedForward(TRAIN, ThreadNum);
		NN->FC_CalGrads(cur_Y, Y_hat, ThreadNum);
		delete cur_Y;
	}
	else if (NetType == LENET1)
	{
	    if(dropConnect)
        {
            for(int jj = 0; jj < numOfLayers - 2; jj++)
            {
                NN->D2[ThreadNum][jj] = new matrix<bool>*[minibatchSize / Cores];
                for(int kk = 0; kk < minibatchSize / Cores; kk++)
                {
                    NN->D2[ThreadNum][jj][kk] = new matrix<bool>(layers[jj + 1].neurons, layers[jj].neurons, Bernoulli, keep_prob[jj + 1]);
                }
            }
        }
		Matrix* cur_X = X->Sub_Mat(0, start, X->Rows() - 1, end);
		Matrix* cur_Y = Y->Sub_Mat(0, start, Y->Rows() - 1, end);
		int m = cur_X->Columns();

		VectVolume AC0 = to_VectorOfVolume(cur_X, 28, 28, 1, m);
		delete cur_X;

		NN->Conv_Cache[ThreadNum].put("AC0", AC0);
		NN->convLayer(TRAIN,1, 1, LEAKYRELU, ThreadNum);           //stride=1, A_index=1,W_index=1
		NN->poolLayer(2, 2, AVG, 1, ThreadNum);         //stride=2, f=2 ,mode ="avg",A_index=1
		NN->convLayer(TRAIN, 1, 2, LEAKYRELU, ThreadNum);           //stride=1 ,A_index=2,W_index=2
		NN->poolLayer(2, 2, AVG, 2, ThreadNum);         //f=5 ,mode ="avg",A_index=2

		VectVolume ACP2 = NN->Conv_Cache[ThreadNum]["ACP2"];
		Matrix* A0 = to_FC(ACP2);
		NN->FC_Cache[ThreadNum].put("A0", A0);
		Matrix* Y_hat = NN->FC_FeedForward(TRAIN, ThreadNum);
		NN->FC_CalGrads(cur_Y, Y_hat, ThreadNum);
		delete cur_Y;

		Matrix* dA0 = NN->FC_Grades[ThreadNum]["dA0"];
		VectVolume dACP2 = to_VectorOfVolume(dA0, ACP2[0][0]->Rows(), ACP2[0][0]->Columns(), ACP2[0].size(), m);

		NN->Conv_Grades[ThreadNum].put("dACP2", dACP2);
		NN->pool_backward(2, 2, AVG, 2, ThreadNum);			   //f=2,stride=2,mode ="avg",A_index=2
		NN->ConvBackwardOptimized(1, 2, LEAKYRELU, ThreadNum);        //stride=1 ,A_index=2,W_index=2
		NN->pool_backward(2, 2, AVG, 1, ThreadNum);			   //f=2,stride=2,mode ="avg",A_index=1
		NN->ConvBackwardOptimized(1, 1, LEAKYRELU, ThreadNum);        //stride=1 ,A_index=1,W_index=1


		NN->FC_Cache[ThreadNum].DeleteThenClear();
		NN->Conv_Cache[ThreadNum].DeleteThenClearObj();

		NN->Conv_Grades[ThreadNum].DeleteThenEraseObj("dACP2");
		NN->Conv_Grades[ThreadNum].DeleteThenEraseObj("dACP1");
		NN->Conv_Grades[ThreadNum].DeleteThenEraseObj("dAC2");
		NN->Conv_Grades[ThreadNum].DeleteThenEraseObj("dAC1");
	}
	else if (NetType == other)
	{
	    if(dropConnect)
        {
            for(int jj = 0; jj < numOfLayers - 2; jj++)
            {
                NN->D2[ThreadNum][jj] = new matrix<bool>*[minibatchSize / Cores];
                for(int kk = 0; kk < minibatchSize / Cores; kk++)
                {
                    NN->D2[ThreadNum][jj][kk] = new matrix<bool>(layers[jj + 1].neurons, layers[jj].neurons, Bernoulli, keep_prob[jj + 1]);
                }
            }
        }
		Matrix* cur_X = X->Sub_Mat(0, start, X->Rows() - 1, end);
		Matrix* cur_Y = Y->Sub_Mat(0, start, Y->Rows() - 1, end);
		int m = cur_X->Columns();

		VectVolume AC0 = to_VectorOfVolume(cur_X, 28, 28, 1, m);
		delete cur_X;

		NN->Conv_Cache[ThreadNum].put("AC0", AC0);
		NN->convLayer(TRAIN,1, 1, LEAKYRELU, ThreadNum);           //stride=1, A_index=1,W_index=1
		NN->poolLayer(2, 2, AVG, 1, ThreadNum);                    //stride=2, f=2 ,mode ="avg",A_index=1


		VectVolume ACP1 = NN->Conv_Cache[ThreadNum]["ACP1"];
		Matrix* A0 = to_FC(ACP1);
		NN->FC_Cache[ThreadNum].put("A0", A0);
		Matrix* Y_hat = NN->FC_FeedForward(TRAIN, ThreadNum);
		NN->FC_CalGrads(cur_Y, Y_hat, ThreadNum);
		delete cur_Y;

		Matrix* dA0 = NN->FC_Grades[ThreadNum]["dA0"];
		VectVolume dACP1 = to_VectorOfVolume(dA0, ACP1[0][0]->Rows(), ACP1[0][0]->Columns(), ACP1[0].size(), m);

		NN->Conv_Grades[ThreadNum].put("dACP1", dACP1);
		NN->pool_backward(2, 2, AVG, 1, ThreadNum);			   //f=2,stride=2,mode ="avg",A_index=1
		NN->ConvBackwardOptimized(1, 1, LEAKYRELU, ThreadNum);        //stride=1 ,A_index=1,W_index=1

		NN->FC_Cache[ThreadNum].DeleteThenClear();
		NN->Conv_Cache[ThreadNum].DeleteThenClearObj();
		NN->Conv_Grades[ThreadNum].DeleteThenEraseObj("dACP1");
		NN->Conv_Grades[ThreadNum].DeleteThenEraseObj("dAC1");
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::Average()
{
    /*ARGUMENT LIST*/
    TypeOfNet NetType = Arg->NetType;
    int numOfLayers = Arg->numOfLayers;
    bool batchNorm = Arg->batchNorm;
    /*END OF ARGUMENT LIST*/

	//get out of Average with FC_Grades[cores] full, others deleted
	Matrix* Matptr;
	if (NetType == FC)
	{
		for (int i = 1; i < numOfLayers; i++)
		{
			Matrix* dW = nullptr;
			Matrix* db = nullptr;
			Matrix* dg1 = nullptr;
			Matrix* dg2 = nullptr;
			Matrix* rv = nullptr;
			Matrix* rm = nullptr;
			dW = FC_Grades[0][CharGen("dW", i)];
			if (batchNorm)
			{
				if (isLastepoch)
				{
					rv = FC_Grades[0][CharGen("rv", i)];
					rm = FC_Grades[0][CharGen("rm", i)];
				}
				dg1 = FC_Grades[0][CharGen("dg1", i)];
				dg2 = FC_Grades[0][CharGen("dg2", i)];
			}
			else
			{
				db = FC_Grades[0][CharGen("db", i)];
			}
			for (int j = 1; j < Cores; j++)
			{
				if (j == 1)
				{
					dW = dW->add(FC_Grades[j][CharGen("dW", i)]);
					if (batchNorm)
					{
						if (isLastepoch)
						{
							rv = rv->add(FC_Grades[j][CharGen("rv", i)]);
							rm = rm->add(FC_Grades[j][CharGen("rm", i)]);
						}
						dg1 = dg1->add(FC_Grades[j][CharGen("dg1", i)]);
						dg2 = dg2->add(FC_Grades[j][CharGen("dg2", i)]);
					}
					else
					{
						db = db->add(FC_Grades[j][CharGen("db", i)]);
					}
				}
				else
				{
					Matptr = dW;
					dW = dW->add(FC_Grades[j][CharGen("dW", i)]);
					delete Matptr;
					if (batchNorm)
					{
						if (isLastepoch)
						{
							Matptr = rv;
							rv = rv->add(FC_Grades[j][CharGen("rv", i)]);
							delete Matptr;

							Matptr = rm;
							rm = rm->add(FC_Grades[j][CharGen("rm", i)]);
							delete Matptr;
						}
						Matptr = dg1;
						dg1 = dg1->add(FC_Grades[j][CharGen("dg1", i)]);
						delete Matptr;

						Matptr = dg2;
						dg2 = dg2->add(FC_Grades[j][CharGen("dg2", i)]);
						delete Matptr;
					}
					else
					{
						Matptr = db;
						db = db->add(FC_Grades[j][CharGen("db", i)]);
						delete Matptr;
					}
				}

			}
			Matptr = dW;
			dW = dW->div(float(Cores));
			delete Matptr;

			FC_Grades[Cores].put(CharGen("dW", i), dW);
			if (batchNorm)
			{
				if (isLastepoch)
				{
					Matptr = rv;
					rv = rv->div(float(Cores));
					delete Matptr;

					Matptr = rm;
					rm = rm->div(float(Cores));
					delete Matptr;

					FC_Parameters.DeleteThenReplace(CharGen("rv", i), rv);
					FC_Parameters.DeleteThenReplace(CharGen("rm", i), rm);
				}
				Matptr = dg1;
				dg1 = dg1->div(float(Cores));
				delete Matptr;

				Matptr = dg2;
				dg2 = dg2->div(float(Cores));
				delete Matptr;

				FC_Grades[Cores].put(CharGen("dg1", i), dg1);
				FC_Grades[Cores].put(CharGen("dg2", i), dg2);
			}
			else
			{
				Matptr = db;
				db = db->div(float(Cores));
				delete Matptr;
				FC_Grades[Cores].put(CharGen("db", i), db);
			}

		}
	}



	else if (NetType == LENET1)
	{
		// Fully-Connected part
		for (int i = 1; i < numOfLayers; i++)
		{
			Matrix* dW = nullptr;
			Matrix* db = nullptr;
			Matrix* dg1 = nullptr;
			Matrix* dg2 = nullptr;

			dW = FC_Grades[0][CharGen("dW", i)];
			if (batchNorm)
			{
				dg1 = FC_Grades[0][CharGen("dg1", i)];
				dg2 = FC_Grades[0][CharGen("dg2", i)];
			}
			else
			{
				db = FC_Grades[0][CharGen("db", i)];
			}
			for (int j = 1; j < Cores; j++)
			{
				if (j == 1)
				{
					dW = dW->add(FC_Grades[j][CharGen("dW", i)]);
					if (batchNorm)
					{
						dg1 = dg1->add(FC_Grades[j][CharGen("dg1", i)]);
						dg2 = dg2->add(FC_Grades[j][CharGen("dg2", i)]);
					}
					else
					{
						db = db->add(FC_Grades[j][CharGen("db", i)]);
					}
				}
				else
				{
					Matptr = dW;
					dW = dW->add(FC_Grades[j][CharGen("dW", i)]);
					delete Matptr;
					if (batchNorm)
					{
						Matptr = dg1;
						dg1 = dg1->add(FC_Grades[j][CharGen("dg1", i)]);
						delete Matptr;

						Matptr = dg2;
						dg2 = dg2->add(FC_Grades[j][CharGen("dg2", i)]);
						delete Matptr;
					}
					else
					{
						Matptr = db;
						db = db->add(FC_Grades[j][CharGen("db", i)]);
						delete Matptr;
					}

				}
			}
			Matptr = dW;
			dW = dW->div(float(Cores));
			delete Matptr;

			FC_Grades[Cores].put(CharGen("dW", i), dW);
			if (batchNorm)
			{
				Matptr = dg1;
				dg1 = dg1->div(float(Cores));
				delete Matptr;

				Matptr = dg2;
				dg2 = dg2->div(float(Cores));
				delete Matptr;

				FC_Grades[Cores].put(CharGen("dg1", i), dg1);
				FC_Grades[Cores].put(CharGen("dg2", i), dg2);
			}
			else
			{
				Matptr = db;
				db = db->div(float(Cores));
				delete Matptr;

				FC_Grades[Cores].put(CharGen("db", i), db);
			}
		}


		// Convolution part
		VectVolume dWC1(dWC1_NoOfFilters, dWC1_Channels, dWC1_f, dWC1_f);

		VectVolume dWC1_temp = Conv_Grades[0]["dWC1"];

		for (int i = 0; i < dWC1_NoOfFilters; i++)
		{
			for (int j = 0; j < dWC1_Channels; j++)
			{
				dWC1[i][j] = dWC1_temp[i][j];
			}
		}

		for (int k = 1; k < Cores; k++)
		{
			VectVolume dWC1_temp = Conv_Grades[k]["dWC1"];
			if (k == 1)
			{
				for (int i = 0; i < dWC1_NoOfFilters; i++)
				{
					for (int j = 0; j < dWC1_Channels; j++)
					{
						dWC1[i][j] = (dWC1[i][j])->add(dWC1_temp[i][j]);
					}
				}
			}
			else
			{
				for (int i = 0; i < dWC1_NoOfFilters; i++)
				{
					for (int j = 0; j < dWC1_Channels; j++)
					{
						Matptr = dWC1[i][j];
						dWC1[i][j] = (dWC1[i][j])->add(dWC1_temp[i][j]);
						delete Matptr;
					}
				}
			}
		}

		for (int i = 0; i < dWC1_NoOfFilters; i++)
		{
			for (int j = 0; j < dWC1_Channels; j++)
			{
				Matptr = dWC1[i][j];
				dWC1[i][j] = (dWC1[i][j])->div(float(Cores));
				delete Matptr;
			}
		}


		VectVolume dWC2(dWC2_NoOfFilters, dWC2_Channels, dWC2_f, dWC2_f);

		VectVolume dWC2_temp = Conv_Grades[0]["dWC2"];
		for (int i = 0; i < dWC2_NoOfFilters; i++)
		{
			for (int j = 0; j < dWC2_Channels; j++)
			{
				dWC2[i][j] = dWC2_temp[i][j];
			}
		}

		for (int k = 1; k < Cores; k++)
		{
			VectVolume dWC2_temp = Conv_Grades[k]["dWC2"];
			if (k == 1)
			{
				for (int i = 0; i < dWC2_NoOfFilters; i++)
				{
					for (int j = 0; j < dWC2_Channels; j++)
					{
						dWC2[i][j] = (dWC2[i][j])->add(dWC2_temp[i][j]);
					}
				}
			}
			else
			{
				for (int i = 0; i < dWC2_NoOfFilters; i++)
				{
					for (int j = 0; j < dWC2_Channels; j++)
					{
						Matptr = dWC2[i][j];
						dWC2[i][j] = (dWC2[i][j])->add(dWC2_temp[i][j]);
						delete Matptr;
					}
				}
			}
		}

		for (int i = 0; i < dWC2_NoOfFilters; i++)
		{
			for (int j = 0; j < dWC2_Channels; j++)
			{
				Matptr = dWC2[i][j];
				dWC2[i][j] = (dWC2[i][j])->div(float(Cores));
				delete Matptr;
			}
		}

		Matrix* dbC1 = nullptr;
		dbC1 = Conv_dbiases[0]["dbC1"];
		for (int i = 1; i < Cores; i++)
		{
			if (i == 1)
			{
				dbC1 = dbC1->add(Conv_dbiases[i]["dbC1"]);
			}
			else
			{
				Matptr = dbC1;
				dbC1 = dbC1->add(Conv_dbiases[i]["dbC1"]);
				delete Matptr;
			}
		}

		Matptr = dbC1;
		dbC1 = dbC1->div(float(Cores));
		delete Matptr;

		Matrix* dbC2 = nullptr;
		dbC2 = Conv_dbiases[0]["dbC2"];

		for (int i = 1; i < Cores; i++)
		{
			if (i == 1)
			{
				dbC2 = dbC2->add(Conv_dbiases[i]["dbC2"]);
			}
			else
			{
				Matptr = dbC2;
				dbC2 = dbC2->add(Conv_dbiases[i]["dbC2"]);
				delete Matptr;
			}
		}
		Matptr = dbC2;
		dbC2 = dbC2->div(float(Cores));
		delete Matptr;

		Conv_Grades[Cores].put("dWC1", dWC1);
		Conv_Grades[Cores].put("dWC2", dWC2);
		Conv_dbiases[Cores].put("dbC1", dbC1);
		Conv_dbiases[Cores].put("dbC2", dbC2);

	}
	else if (NetType == other)
	{
		// Fully-Connected part
		for (int i = 1; i < numOfLayers; i++)
		{
			Matrix* dW = nullptr;
			Matrix* db = nullptr;
			Matrix* dg1 = nullptr;
			Matrix* dg2 = nullptr;

			dW = FC_Grades[0][CharGen("dW", i)];
			if (batchNorm)
			{
				dg1 = FC_Grades[0][CharGen("dg1", i)];
				dg2 = FC_Grades[0][CharGen("dg2", i)];
			}
			else
			{
				db = FC_Grades[0][CharGen("db", i)];
			}
			for (int j = 1; j < Cores; j++)
			{
				if (j == 1)
				{
					dW = dW->add(FC_Grades[j][CharGen("dW", i)]);
					if (batchNorm)
					{
						dg1 = dg1->add(FC_Grades[j][CharGen("dg1", i)]);
						dg2 = dg2->add(FC_Grades[j][CharGen("dg2", i)]);
					}
					else
					{
						db = db->add(FC_Grades[j][CharGen("db", i)]);
					}
				}
				else
				{
					Matptr = dW;
					dW = dW->add(FC_Grades[j][CharGen("dW", i)]);
					delete Matptr;
					if (batchNorm)
					{
						Matptr = dg1;
						dg1 = dg1->add(FC_Grades[j][CharGen("dg1", i)]);
						delete Matptr;

						Matptr = dg2;
						dg2 = dg2->add(FC_Grades[j][CharGen("dg2", i)]);
						delete Matptr;
					}
					else
					{
						Matptr = db;
						db = db->add(FC_Grades[j][CharGen("db", i)]);
						delete Matptr;
					}

				}
			}
			Matptr = dW;
			dW = dW->div(float(Cores));
			delete Matptr;

			FC_Grades[Cores].put(CharGen("dW", i), dW);
			if (batchNorm)
			{
				Matptr = dg1;
				dg1 = dg1->div(float(Cores));
				delete Matptr;

				Matptr = dg2;
				dg2 = dg2->div(float(Cores));
				delete Matptr;

				FC_Grades[Cores].put(CharGen("dg1", i), dg1);
				FC_Grades[Cores].put(CharGen("dg2", i), dg2);
			}
			else
			{
				Matptr = db;
				db = db->div(float(Cores));
				delete Matptr;

				FC_Grades[Cores].put(CharGen("db", i), db);
			}
		}


		// Convolution part
		VectVolume dWC1(dWC1_NoOfFilters, dWC1_Channels, dWC1_f, dWC1_f);

		VectVolume dWC1_temp = Conv_Grades[0]["dWC1"];

		for (int i = 0; i < dWC1_NoOfFilters; i++)
		{
			for (int j = 0; j < dWC1_Channels; j++)
			{
				dWC1[i][j] = dWC1_temp[i][j];
			}
		}

		for (int k = 1; k < Cores; k++)
		{
			VectVolume dWC1_temp = Conv_Grades[k]["dWC1"];
			if (k == 1)
			{
				for (int i = 0; i < dWC1_NoOfFilters; i++)
				{
					for (int j = 0; j < dWC1_Channels; j++)
					{
						dWC1[i][j] = (dWC1[i][j])->add(dWC1_temp[i][j]);
					}
				}
			}
			else
			{
				for (int i = 0; i < dWC1_NoOfFilters; i++)
				{
					for (int j = 0; j < dWC1_Channels; j++)
					{
						Matptr = dWC1[i][j];
						dWC1[i][j] = (dWC1[i][j])->add(dWC1_temp[i][j]);
						delete Matptr;
					}
				}
			}
		}

		for (int i = 0; i < dWC1_NoOfFilters; i++)
		{
			for (int j = 0; j < dWC1_Channels; j++)
			{
				Matptr = dWC1[i][j];
				dWC1[i][j] = (dWC1[i][j])->div(float(Cores));
				delete Matptr;
			}
		}

		Matrix* dbC1 = nullptr;
		dbC1 = Conv_dbiases[0]["dbC1"];
		for (int i = 1; i < Cores; i++)
		{
			if (i == 1)
			{
				dbC1 = dbC1->add(Conv_dbiases[i]["dbC1"]);
			}
			else
			{
				Matptr = dbC1;
				dbC1 = dbC1->add(Conv_dbiases[i]["dbC1"]);
				delete Matptr;
			}
		}

		Matptr = dbC1;
		dbC1 = dbC1->div(float(Cores));
		delete Matptr;

		Conv_Grades[Cores].put("dWC1", dWC1);
		Conv_dbiases[Cores].put("dbC1", dbC1);
	}
}
