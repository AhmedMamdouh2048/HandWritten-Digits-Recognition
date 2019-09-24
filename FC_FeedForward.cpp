#include "NeuralNetwork.h"
Matrix* NeuralNetwork::FC_FeedForward(Mode mode, int ThreadNum)
{
    /*ARGUMENT LIST*/
    int L = Arg->numOfLayers;
	bool batchNorm = Arg->batchNorm;
	bool dropout = Arg->dropout;
	bool dropConnect = Arg->dropConnect;
	bool BatchMultiThread = Arg->BatchMultiThread;
	layer* layers = Arg->layers;
	float* keep_prob = Arg->keep_prob;
    /*END OF ARGUMENT LIST*/

    /*INITIALIZATION OF PARAMETERS OF BATCHNORM*/
	Matrix*  g1 = nullptr;					//gamma for each layer
	Matrix*  g2 = nullptr;					//beta for each layer
	Matrix** mean = nullptr;                //mean of z for each layer
	Matrix** var = nullptr;                 //standard deviation of z for each layer
	Matrix*  zmeu = nullptr;				    //z-mean of z
	Matrix*  z_telda = nullptr;				//(z-mean)/varience of z
	Matrix*  z_new = nullptr;					//z after normalization,scaling and shifting by gamma and beta
	float    eps = 1e-7;                    //to make sure that we don`t divide by zero
	float    beta = 0.9;
	if(mode == TRAIN)
    {
        mean = new Matrix*[L - 1];     //mean of z for each layer
        var = new Matrix*[L - 1];      //standard deviation of z for each layer
    }
	/*END OF INITIALIZATION OF PARAMETERS OF BATCHNORM*/


	/*TEMPORARY POINTERS*/
	Matrix*  z = nullptr;
	Matrix*  A = nullptr;
    Matrix*  MatPtr = nullptr;
    Matrix*  temp1 = nullptr;
    Matrix*  temp2 = nullptr;
    Matrix* b=nullptr;
    /*END OF TEMPORARY POINTERS*/


    /*BEGINNING OF THE FEEDFORWARD*/
	for (int i = 0; i < L - 1; i++)
	{
		Matrix* W = FC_Parameters[CharGen("W", i + 1)];
		if(!batchNorm)
            b = FC_Parameters[CharGen("b", i + 1)];
		Matrix* Aprev = FC_Cache[ThreadNum][CharGen("A", i)];

		if (batchNorm)
		{
		    if(dropConnect && i != L - 2 && mode == TRAIN)
            {
                //z=(W*D2).dot(Aprev)+b;
                z = new Matrix(W->Rows(), Aprev->Columns());
                for(int ii = 0; ii < Aprev->Columns(); ii++)
                {
                    MatPtr = W->MultBool(D2[ThreadNum][i][ii]);
                    for(int jj = 0; jj < W->Rows(); jj++)
                    {
                        float sum = 0;
                        for(int kk = 0; kk < W->Columns(); kk++)
                        {
                            sum += MatPtr->access(jj, kk) * Aprev->access(kk, ii);
                        }
                        z->access(jj, ii) = sum;
                    }
                    delete MatPtr;
                }
                MatPtr = z;
                z = z->div(keep_prob[i + 1]);
                delete MatPtr;
            }
            else
            {
                z = DOT(W,Aprev);

            }

			if (mode == TRAIN)
			{
			    //*mean[i] = z.sum("column") / z.Columns();
                temp1=z->SUM("column");
				mean[i] = temp1 ->div(z->Columns());
				delete temp1;

				zmeu = z->sub(mean[i]);

				//*var[i]=(zmeu.square()).sum("column") / z.Columns();
				temp1=zmeu->SQUARE();
				temp2=temp1->SUM("column");
				var[i] = temp2 ->div(z->Columns());
				delete temp1;
				delete temp2;

				//z_telda = zmeu / (*var[i]+eps).Sqrt();
                temp1=var[i]->add(eps);
                temp2=temp1->SQRT();
				z_telda = zmeu->div(temp2);
				delete temp1;
				delete temp2;

				if (isLastepoch)
				{
					Matrix* r_mean = FC_Parameters[CharGen("rm", i + 1)];
					//r_mean=r_mean*beta+(*mean[i])*(1-beta);
					temp1=r_mean->mul(beta);
					temp2=mean[i]->mul(1 - beta);
					r_mean = temp1->add(temp2);
					delete temp1;
                    delete temp2;

                    if(BatchMultiThread)
                    {
                        if(!FC_Grades[ThreadNum].exist(CharGen("rm", i + 1)))
                        {
                            FC_Grades[ThreadNum].put(CharGen("rm", i + 1), r_mean);
                        }
                        else
                        {
                            FC_Grades[ThreadNum].DeleteThenReplace(CharGen("rm", i + 1), r_mean);
                        }
                    }
                    else
                    {
                        FC_Parameters.DeleteThenReplace(CharGen("rm", i + 1), r_mean);
                    }

					//rr_var=r_var*beta+(*var[i])*(1-beta);
					Matrix* r_var = FC_Parameters[CharGen("rv", i + 1)];
					temp1=r_var->mul(beta);
					temp2=var[i]->mul(1 - beta);
					r_var = temp1->add(temp2);
					delete temp1;
                    delete temp2;

                    if(BatchMultiThread)
                    {
                        if(!FC_Grades[ThreadNum].exist(CharGen("rv", i + 1)))
                        {
                            FC_Grades[ThreadNum].put(CharGen("rv", i + 1), r_var);
                        }
                        else
                        {
                            FC_Grades[ThreadNum].DeleteThenReplace(CharGen("rv", i + 1), r_var);
                        }
                    }
                    else
                    {
                        FC_Parameters.DeleteThenReplace(CharGen("rv", i + 1), r_var);
                    }
				}
			}
			else
			{
			    /*TEST*/
				Matrix* r_mean = FC_Parameters[CharGen("rm", i + 1)];
				Matrix* r_var = FC_Parameters[CharGen("rv", i + 1)];
				zmeu = z->sub(r_mean);

				//z_telda = zmeu / (r_var+eps).Sqrt();
				temp1=r_var->add(eps);
				temp2=temp1->SQRT();
				z_telda = zmeu->div(temp2);
				delete temp1;
                delete temp2;
			}
            delete z;

			g1 = FC_Parameters[CharGen("g1", i + 1)];
			g2 = FC_Parameters[CharGen("g2", i + 1)];

			//z_new=z_telda*g1+g2
			temp1=z_telda->mul(g1);
			z_new = temp1->add(g2);
			delete temp1;

			if (mode == TRAIN)
			{
				FC_Cache[ThreadNum].put(CharGen("zm", i + 1), zmeu);
				FC_Cache[ThreadNum].put(CharGen("zt", i + 1), z_telda);
				FC_Cache[ThreadNum].put(CharGen("zn", i + 1), z_new);
				FC_Cache[ThreadNum].put(CharGen("m", i + 1), mean[i]);
				FC_Cache[ThreadNum].put(CharGen("var", i + 1), var[i]);
			}
			else
            {
                delete zmeu;
                delete z_telda;
            }
		}
		else
		{
		    if(dropConnect && i != L - 2 && mode == TRAIN)
            {
                //z=(W*D2).dot(Aprev)+b;
                temp1 = new Matrix(W->Rows(), Aprev->Columns());
                for(int ii = 0; ii < Aprev->Columns(); ii++)
                {
                    //Matptr = W*D2
                    MatPtr = W->MultBool(D2[ThreadNum][i][ii]);
                    //z = Matptr.dot(Aprev)
                    for(int jj = 0; jj < W->Rows(); jj++)
                    {
                        float sum = 0;
                        for(int kk = 0; kk < W->Columns(); kk++)
                        {
                            sum += MatPtr->access(jj, kk) * Aprev->access(kk, ii);
                        }
                        temp1->access(jj, ii) = sum;
                    }
                    delete MatPtr;
                }
                MatPtr = temp1;
                temp1 = temp1->div(keep_prob[i + 1]);
                delete MatPtr;
                z=temp1->add(b);
                delete temp1;
                FC_Cache[ThreadNum].put(CharGen("z", i + 1), z);
            }
            else
            {
                //z=W.dot(Aprev)+b;
                temp1=DOT(W,Aprev);
                z = temp1->add(b);
                delete temp1;
                FC_Cache[ThreadNum].put(CharGen("z", i + 1), z);
            }
		}

		ActivationType activation = layers[i + 1].activation;

		if(batchNorm)
        {
            A=activ(z_new,activation);
            if(mode == TEST)
                delete z_new;
        }
        else
            A=activ(z,activation);

		if (dropout && mode == TRAIN)
		{
			// A=A * (*(D[i+1]));
			// A=A/keep_prob[i+1];
			MatPtr=A;
			temp1=A->mul(D[ThreadNum][i + 1]);
			A=temp1->div(keep_prob[i + 1]);
			delete MatPtr;
			delete temp1;
		}

		FC_Cache[ThreadNum].put(CharGen("A", i + 1), A);
	}
	/*END OF FEEDFORWARD*/

	return A; //yhat
}
