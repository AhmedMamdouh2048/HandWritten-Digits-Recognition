#include "NeuralNetwork.h"
void NeuralNetwork::Conv_BN_feedforward(Mode mode,int index,int ThreadNum)
{
	VectVolume ZCold = Conv_Cache[ThreadNum][CharGen("ZC",index)];
	int m = ZCold.size();
	int c = ZCold[0].size();     //num of channel
	int h = ZCold[0][0]->Rows();
	int w = ZCold[0][0]->Columns();

	VectVolume ZCmeu(m, c);
	VectVolume ZCtelda(m, c);
	VectVolume ZCnew(m, c);

	Matrix*  g1 = nullptr;					//gamma for each layer
	Matrix*  g2 = nullptr;					//beta for each layer
	Matrix* mean = nullptr;                //mean of z for each layer
	Matrix* var = nullptr;                 //standard deviation of z for each layer
	float    eps = 1e-7;                    //to make sure that we don`t divide by zero
	float    beta = 0.9;
	if (mode == TRAIN)
	{
		mean = new Matrix(c,1);     //mean of z for each layer
		var = new Matrix (c, 1);      //standard deviation of z for each layer
	}

	Matrix*  temp1 = nullptr;
	Matrix*  temp2 = nullptr;
	bool BatchMultiThread = Arg->BatchMultiThread;

	if (mode == TRAIN)
	{
		//Calculating Mean
		float mean1 = 0;
		for (int i = 0; i < c; i++)
		{
			for (int j = 0; j < m; j++)
			{
				mean1 += ZCold[j][i]->sumall();
			}
			mean->access(i, 0) = mean1 / (w*h*m);
			mean1 = 0;
		}
		//Calculating Zmeu
		for (int i = 0; i < c; i++)
		{
			mean1 = mean->access(i, 0);
			for (int j = 0; j < m; j++)
			{
				ZCmeu[j][i] = ZCold[j][i]->sub(mean1);
			}
		}
		//Calculating var
		float var1 = 0;
		for (int i = 0; i < c; i++)
		{
			for (int j = 0; j < m; j++)
			{
				temp1 = ZCmeu[j][i]->SQUARE();
				var1 += temp1->sumall();
				delete temp1;
			}
			var->access(i, 0) = var1 / (m*h*w);
		}
		//Calculating Ztelda
		for (int i = 0; i < c; i++)
		{
			var1 = var->access(i, 0);
			var1 = sqrt(var1 + eps);
			for (int j = 0; j < m; j++)
			{
				ZCtelda[j][i] = ZCmeu[j][i]->div(var1);
			}
		}

		if (isLastepoch)
		{
			//r_mean=r_mean*beta+(*mean[i])*(1-beta);
			Matrix* r_mean = Conv_biases[CharGen("rmC", index)];
			temp1 = r_mean->mul(beta);
			temp2 = mean->mul(1 - beta);
			r_mean = temp1->add(temp2);
			delete temp1;
			delete temp2;
			if (BatchMultiThread)
			{
				if (!Conv_dbiases[ThreadNum].exist(CharGen("rmC", index)))
				{
					Conv_dbiases[ThreadNum].put(CharGen("rmC", index), r_mean);
				}
				else
				{
					Conv_dbiases[ThreadNum].DeleteThenReplace(CharGen("rmC", index), r_mean);
				}
			}
			else
			{
				Conv_biases.DeleteThenReplace(CharGen("rmC", index), r_mean);
			}

			//rr_var=r_var*beta+(*var[i])*(1-beta);
			Matrix* r_var = Conv_biases[CharGen("rvC", index)];
			temp1 = r_var->mul(beta);
			temp2 = var->mul(1 - beta);
			r_var = temp1->add(temp2);
			delete temp1;
			delete temp2;
			if (BatchMultiThread)
			{
				if (!Conv_dbiases[ThreadNum].exist(CharGen("rvC", index)))
				{
					Conv_dbiases[ThreadNum].put(CharGen("rvC", index), r_var);
				}
				else
				{
					Conv_dbiases[ThreadNum].DeleteThenReplace(CharGen("rvC", index), r_var);
				}
			}
			else
			{
				Conv_biases.DeleteThenReplace(CharGen("rvC", index), r_var);
			}
		}
	}
	else   //Mode=TEST
	{
		Matrix* r_mean = Conv_biases[CharGen("rmC", index)];
		Matrix* r_var = Conv_biases[CharGen("rvC", index)];
		//Calculating Zmeu
		for (int i = 0; i < c; i++)
		{
			float mean1 = r_mean->access(i, 0);
			for (int j = 0; j < m; j++)
			{
				ZCmeu[j][i] = ZCold[j][i]->sub(mean1);
			}
		}
		//Calculating Ztelda
		for (int i = 0; i < c; i++)
		{
			float var1 = r_var->access(i, 0);
			var1 = sqrt(var1 + eps);
			for (int j = 0; j < m; j++)
			{
				ZCtelda[j][i] = ZCmeu[j][i]->div(var1);
			}
		}
	}
	Conv_Cache[ThreadNum].DeleteThenEraseObj(CharGen("ZC", index));

	g1 = Conv_biases[CharGen("gC1", index)];
	g2 = Conv_biases[CharGen("gC2", index)];
	//z_new=z_telda*g1+g2
	for (int i = 0; i < c; i++)
	{
		float gamma1 = g1->access(i, 0);
		float gamma2 = g2->access(i, 0);
		for (int j = 0; j < m; j++)
		{
			temp1 = ZCtelda[j][i]->mul(gamma1);
			ZCnew[j][i] = temp1->add(gamma2);
			delete temp1;
		}
	}
	if (mode == TRAIN)
	{
		Conv_Cache[ThreadNum].put(CharGen("ZCm", index), ZCmeu);
		Conv_Cache[ThreadNum].put(CharGen("ZCt", index), ZCtelda);
		Conv_Cache[ThreadNum].put(CharGen("ZCn", index), ZCnew);
		Conv_Cache_Mat[ThreadNum].put(CharGen("mC", index), mean);
		Conv_Cache_Mat[ThreadNum].put(CharGen("varC", index), var);
	}
	else
	{
		Conv_Cache[ThreadNum].put(CharGen("ZCn", index), ZCnew);
		ZCmeu.DELETE();
	    ZCtelda.DELETE();
	}

}
///////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::Conv_BN_backprop(VectVolume dZC,int index,int ThreadNum)
{
	int m = dZC.size();
	int c = dZC[0].size();
	int h = dZC[0][0]->Rows();
	int w = dZC[0][0]->Columns();
	Matrix* dgC1 = new Matrix(c, 1);
	Matrix* dgC2 = new Matrix(c, 1);
	Matrix* divar = new Matrix(c, 1);
	Matrix* dmeu = new Matrix(c, 1);
	VectVolume dZCtelda(m, c);
	float temp1 = 0;
	float temp2 = 0;
	float temp3 = 0;
	float eps = 1e-7;

	Matrix* gC1 = Conv_biases[CharGen("gC1", index)];
	Matrix* var = Conv_Cache_Mat[ThreadNum][CharGen("varC", index)];
	VectVolume ZCtelda = Conv_Cache[ThreadNum][CharGen("ZCt", index)];
	VectVolume ZCmeu = Conv_Cache[ThreadNum][CharGen("ZCm", index)];

	/*TEMPORARY POINTERS*/
	Matrix* MatPtr1 = nullptr;
	Matrix* MatPtr2 = nullptr;
	/*END OF TEMPORARY POINTERS*/

	/*getting dgamma and dbeta*/
	/*dzlast here means dzlast_new after normalizing*/

	//dg1 = (dzLast*z_telda).sum("column")
	//*dg2 = dzLast.sum("column")
	for (int i = 0; i < c; i++)
	{
		for (int j = 0; j < m; j++)
		{
			MatPtr1 = dZC[j][i]->mul(ZCtelda[j][i]);
			temp1 += MatPtr1->sumall();
			delete MatPtr1;

			temp2 += dZC[j][i]->sumall();
		}
		dgC1->access(i, 0) = temp1;
		dgC1->access(i, 0) = temp2;
		temp1 = 0;
		temp2 = 0;
	}
	//////////////////////////////////////////////////////
	/*getting dz_telda*/
	//dz_telda = dzLast * (g1)
	for (int i = 0; i < c; i++)
	{
		int g1 = gC1->access(i, 0);
		for (int j = 0; j < m; j++)
		{
			dZCtelda[j][i]= dZC[j][i]->mul(g1);
		}
	}
	//////////////////////////////////////////////////////
	/*getting dvariance*/
	//divar = (dz_telda*zmeu).sum("column")
	temp1 = 0;
	for (int i = 0; i < c; i++)
	{
		for (int j = 0; j < m; j++)
		{
			MatPtr1 = dZCtelda[j][i]->mul(ZCmeu[j][i]);
			temp1 += MatPtr1->sumall();
			delete MatPtr1;
		}
		divar->access(i, 0) = temp1;
		temp1 = 0;
	}
	//dsqrtvar = divar / (var + eps);
	MatPtr1 = var->add(eps);
	Matrix* dsqrtvar = divar->div(MatPtr1);
	delete MatPtr1;
	//t = (var + eps).Sqrt();
	MatPtr1 = var->add(eps);
	Matrix* t = MatPtr1->SQRT();
	delete MatPtr1;
	//Matrix dvar = (dsqrtvar*-0.5) / t;
	MatPtr1 = dsqrtvar->mul(-0.5);
	Matrix* dvar = MatPtr1->div(t);
	delete MatPtr1;
	//////////////////////////////////////////////////////
	/*getting dmeu*/

	//Matrix dmeu1 = (dz_telda*-1) / t;
	//dmeu2 = (zmeu*-2)*dvar;
	//dmeu2 = (dmeu2.sum("column")) / m;
	//dmeu = dmeu1 + dmeu2;
	float dmeu1 = 0;
	float dmeu2 = 0;
	for (int i = 0; i < c; i++)
	{
		temp1 = -1 / t->access(i, 0);
		temp2 = -2 * dvar->access(i, 0);
		for (int j = 0; j < m; j++)
		{
			MatPtr1 = dZCtelda[j][i]->mul(temp1);
			dmeu1 += MatPtr1->sumall();
			delete MatPtr1;

			MatPtr2 = ZCmeu[j][i]->mul(temp2);
			dmeu2 += MatPtr2->sumall();
			delete MatPtr2;
		}
		dmeu2 /= (m*h*w);
		dmeu->access(i, 0) = dmeu1+dmeu2;
		dmeu1 = 0;
		dmeu2 = 0;
	}
	/*getting dzlast (dout) for the incoming layer*/
	//dzLast = dz_telda / t;
	//dzLast = dzLast + (zmeu*dvar)*(2 / m);
	//dzLast = dzLast + dmeu / m;
	for (int i = 0; i < c; i++)
	{
		temp1 = t->access(i, 0);
		temp2 = (2/(h*w*m)) * dvar->access(i, 0);
		temp3 = dmeu->access(i, 0) / (h*w*m);
		for (int j = 0; j < m; j++)
		{
			MatPtr1 = dZC[j][i];
			dZC[j][i]= dZCtelda[j][i]->div(temp1);
			delete MatPtr1;

			MatPtr1 = dZC[j][i];
			MatPtr2 = ZCmeu[j][i]->mul(temp2);
			dZC[j][i] = dZC[j][i]->add(MatPtr2);
			delete MatPtr1;
			delete MatPtr2;

			MatPtr1 = dZC[j][i];
			dZC[j][i] = dZC[j][i]->add(temp3);
			delete MatPtr1;
		}
	}
	Conv_dbiases[ThreadNum].put(CharGen("dgC1", index), dgC1);
	Conv_dbiases[ThreadNum].put(CharGen("dgC2", index), dgC2);
	dZCtelda.DELETE();
	delete divar;
	delete dsqrtvar;
	delete t;
	delete dvar;
	delete dmeu;
}
