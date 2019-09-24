#include "NeuralNetwork.h"

Matrix* create_mask_from_window(Matrix* x)
{
   int indexi=0;
   int indexj=0;
   float maximum=x->access(0,0);
   for(int i=0; i<x->Rows(); i++)
   {
       for(int j=0; j<x->Columns(); j++)
       {
           if(x->access(i,j) > maximum)
           {
              maximum = x->access(i,j);
              indexi = i;
              indexj = j;
           }
       }
   }
   Matrix* mask = new Matrix(x->Rows(),x->Columns(),0);
   mask->access(indexi,indexj)=1;
   return mask;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* distribute_value(float dz,int nh,int nw)
{
    float average=dz/(nh*nw);
    Matrix* a = new Matrix(nh,nw,average);
    return a;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::pool_backward(int f,int stride,Mode mode, int A_index, int ThreadNum)
{
	VectVolume dA=Conv_Grades[ThreadNum][CharGen("dACP",A_index)];
	VectVolume Aprev=Conv_Cache[ThreadNum][CharGen("AC",A_index)];

    int m=Aprev.size();
	int nc=dA[0].size();
    int nh=dA[0][0]->Rows();
    int nw=dA[0][0]->Columns();
    int nc_prev=Aprev[0].size();
    int nh_prev=Aprev[0][0]->Rows();
    int nw_prev=Aprev[0][0]->Columns();

	VectVolume dAprev(m,nc_prev,nh_prev,nw_prev);

	for(int i=0; i<m; i++)
    {
		for (int c = 0; c < nc; c++)
		{
			for (int h = 0; h < nh; h++)
			{
				for (int w = 0; w < nw; w++)
				{
					int vert_start = h * stride;
					int vert_end = vert_start + f;
					int horz_start = w * stride;
					int horz_end = horz_start + f;

					if (mode == MAX)
					{
						Matrix* aprev_slice = Aprev[i][c]->SubMat(vert_start, horz_start, vert_end - 1, horz_end - 1);

						Matrix* tempMask = create_mask_from_window(aprev_slice);

						Matrix* mask = tempMask->mul(dA[i][c]->access(h, w));

						int k = 0; int kk = 0;
						for (int ii = vert_start; ii < vert_end; ii++)
						{
							for (int jj = horz_start; jj < horz_end; jj++)
							{
								dAprev[i][c]->access(ii, jj) = dAprev[i][c]->access(ii, jj) + mask->access(k, kk);
								kk++;
							}
							k++; kk = 0;
						}

						delete aprev_slice;
						delete tempMask;
						delete mask;
					}
					else if (mode == AVG)
					{
						float avg = dA[i][c]->access(h, w) / (f * f);

						for (int ii = vert_start; ii < vert_end; ii++)
						{
							for (int jj = horz_start; jj < horz_end; jj++)
							{
								dAprev[i][c]->access(ii, jj) = dAprev[i][c]->access(ii, jj) + avg;
							}
						}
					}
				}
			}
		}
    }
    Conv_Grades[ThreadNum].put(CharGen("dAC",A_index),dAprev);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::Conv_updateparameters (int iteration, int W_index, int ThreadNum)
{
    /*ARGUMENT LIST*/
    float alpha = Arg->learingRate;
    Optimizer optimizer = Arg->optimizer;
	bool batchNorm = Arg->batchNorm;
    /*END OF ARGUMENT LIST*/

	/*TEMPORARY POINTERS*/
	Matrix* Matptr1 = nullptr;
	Matrix* Matptr2 = nullptr;
	Matrix* Matptr = nullptr;
	Matrix* temp1 = nullptr;
	Matrix* temp2 = nullptr;
	Matrix* temp3 = nullptr;
	Matrix* temp4 = nullptr;
	/*END OF TEMPORARY POINTERS*/

	//filters(W) & b will be in dictionary parameters..dW &db will be in dictionary grades
    VectVolume WC = Conv_Weights[CharGen("WC", W_index)];
	VectVolume dWC = Conv_Grades[ThreadNum][CharGen("dWC", W_index)];
	Matrix* bC = nullptr;
	Matrix* dbC = nullptr;
	if(!batchNorm)
    {
        bC = Conv_biases[CharGen("bC", W_index)];
        dbC = Conv_dbiases[ThreadNum][CharGen("dbC", W_index)];
    }

	/*START OF GRADIENT DESCENT OPTIMIZER*/
	if (optimizer == GRADIENT_DESCENT)
	{
		for (int i = 0; i < WC.size(); i++)
		{
			for (int j = 0; j < WC[0].size(); j++)
			{
				//WC[i][j] = WC[i][j] - dWC[i][j] * alpha;
				Matptr1 = WC[i][j];
				Matptr2 = dWC[i][j]->mul(alpha);
				WC[i][j] = Matptr1->sub(Matptr2);
				delete Matptr1;
				delete Matptr2;
			}
		}

		if (!batchNorm)
		{
			//bC = bC - dbC * alpha;
			Matptr1 = bC;
			Matptr2 = dbC->mul(alpha);
			bC = Matptr1->sub(Matptr2);
			Conv_biases.replace(CharGen("bC", W_index), bC);
			delete Matptr1;
			delete Matptr2;
		}
		else
		{
			//g1 = g1 - dg1 * alpha;
			Matrix* gC1 = Conv_biases[CharGen("gC1", W_index)];
			Matrix* dgC1 = Conv_dbiases[ThreadNum][CharGen("dgC1", W_index)];
			Matptr = gC1;
			temp1 = dgC1->mul(alpha);
			gC1 = gC1->sub(temp1);
			delete Matptr;
			delete temp1;
			Conv_biases.replace(CharGen("gC1", W_index), gC1);
			Conv_dbiases[ThreadNum].DeleteThenErase(CharGen("dgC1", W_index));
			//g2 = g2 - dg2 * alpha;
			Matrix* gC2 = Conv_biases[CharGen("gC2", W_index)];
			Matrix* dgC2 = Conv_dbiases[ThreadNum][CharGen("dg2", W_index)];
			Matptr = gC2;
			temp1 = dgC2->mul(alpha);
			gC2 = gC2->sub(temp1);
			delete Matptr;
			delete temp1;
			Conv_biases.replace(CharGen("gC2", W_index), gC2);
			Conv_dbiases[ThreadNum].DeleteThenErase(CharGen("dgC2", W_index));
		}
	}
	/*END OF GRADIENT DESCENT OPTIMIZER*/

	/*START OF ADAM OPTIMIZER*/
	else if (optimizer == ADAM)
	{
		float beta1 = 0.9;
		float beta2 = 0.999;
		float epsilon = 1e-8;
		Matrix* VdbC = nullptr;
		Matrix* SdbC = nullptr;
		VectVolume VdwC = ADAM_dWC[CharGen("VdwC", W_index)];
		VectVolume SdwC = ADAM_dWC[CharGen("SdwC", W_index)];
		if (!batchNorm)
		{
		    VdbC = ADAM_dbC[CharGen("VdbC", W_index)];
		    SdbC = ADAM_dbC[CharGen("SdbC", W_index)];
		}

		/* Updating VdwC, SdwC */
		for (int i = 0; i < VdwC.size(); i++)
		{
			for (int j = 0; j < VdwC[0].size(); j++)
			{
				//VdwC[i][j] = (VdwC[i][j] * (beta1 * momentum)) + (dWC[i][j] * (1 - beta1 * momentum));
				Matptr = VdwC[i][j];
				temp1 = VdwC[i][j]->mul(beta1 * momentum);
				temp2 = dWC[i][j]->mul(1 - beta1 * momentum);
				VdwC[i][j] = temp1->add(temp2);
				delete Matptr;
				delete temp1;
				delete temp2;

				//SdwC[i][j] = (SdwC[i][j] * beta2) + (dWC[i][j].square() * (1 - beta2));
				Matptr = SdwC[i][j];
				temp1 = SdwC[i][j]->mul(beta2);
				temp2 = dWC[i][j]->SQUARE();
				temp3 = temp2->mul(1 - beta2);
				SdwC[i][j] = temp1->add(temp3);
				delete Matptr;
				delete temp1;
				delete temp2;
				delete temp3;
			}
		}

		if (!batchNorm)
		{
			/* Updating VdbC, SdbC */
			//VdbC = (VdbC * (beta1 * momentum)) + (dbC * (1 - beta1 * momentum));
			Matptr = VdbC;
			temp1 = VdbC->mul(beta1 * momentum);
			temp2 = dbC->mul(1 - beta1 * momentum);
			VdbC = temp1->add(temp2);
			delete Matptr;
			delete temp1;
			delete temp2;
			ADAM_dbC.replace(CharGen("VdbC", W_index), VdbC);

			//SdbC = (SdbC * beta2) + (dbC.square() * (1 - beta2));
			Matptr = SdbC;
			temp1 = SdbC->mul(beta2);
			temp2 = dbC->SQUARE();
			temp3 = temp2->mul(1 - beta2);
			SdbC = temp1->add(temp3);
			delete Matptr;
			delete temp1;
			delete temp2;
			delete temp3;
			ADAM_dbC.replace(CharGen("SdbC", W_index), SdbC);
		}
		/* Correcting first iterations */
		VectVolume VdwC_corr(WC.size(), WC[0].size());
		VectVolume SdwC_corr(WC.size(), WC[0].size());
		Matrix* VdbC_corr = nullptr;
		Matrix* SdbC_corr = nullptr;
		for (int i = 0; i < VdwC_corr.size(); i++)
		{
			for (int j = 0; j < VdwC_corr[0].size(); j++)
			{
				VdwC_corr[i][j] = VdwC[i][j]->div(1 - pow(beta1, iteration + 1));
				SdwC_corr[i][j] = SdwC[i][j]->div(1 - pow(beta2, iteration + 1));
			}
		}

		if (!batchNorm)
		{
			VdbC_corr = VdbC->div(1 - pow(beta1, iteration + 1));
			SdbC_corr = SdbC->div(1 - pow(beta2, iteration + 1));
		}

		/* Updating Parameters */
		for (int i = 0; i < WC.size(); i++)
		{
			for (int j = 0; j < WC[0].size(); j++)
			{
				//temp = VdwC[i][j]_corr / (SdwC[i][j]_corr.Sqrt() + epsilon);
				//WC[i][j] = WC[i][j] - temp * alpha;
				Matptr = WC[i][j];
				temp1 = SdwC_corr[i][j]->SQRT();
				temp2 = temp1->add(epsilon);
				temp3 = VdwC_corr[i][j]->div(temp2);
				temp4 = temp3->mul(alpha);
				WC[i][j] = WC[i][j]->sub(temp4);
				delete Matptr;
				delete temp1;
				delete temp2;
				delete temp3;
				delete temp4;

			}
		}
		VdwC_corr.DELETE();
		SdwC_corr.DELETE();

		if (!batchNorm)
		{
			//Matrix temp = VdbC_corr / (SdbC_corr.Sqrt() + epsilon);
			//Matrix buC = bC - temp * alpha;
			Matptr = bC;
			temp1 = SdbC_corr->SQRT();
			temp2 = temp1->add(epsilon);
			temp3 = VdbC_corr->div(temp2);
			temp4 = temp3->mul(alpha);
			bC = bC->sub(temp4);
			delete Matptr;
			delete temp1;
			delete temp2;
			delete temp3;
			delete temp4;
			Conv_biases.replace(CharGen("bC", W_index), bC);
			delete VdbC_corr;
			delete SdbC_corr;
		}
		else
		{
			/*Getting variables from dictionaries*/
			Matrix* vdgC1 = ADAM_dbC[CharGen("vdg1C", W_index)];
			Matrix* sdgC1 = ADAM_dbC[CharGen("sdg1C", W_index)];
			Matrix* vdgC2 = ADAM_dbC[CharGen("vdg2C", W_index)];
			Matrix* sdgC2 = ADAM_dbC[CharGen("sdg2C", W_index)];


			Matrix* dgC1 = Conv_dbiases[ThreadNum][CharGen("dgC1", W_index)];
			Matrix* dgC2 = Conv_dbiases[ThreadNum][CharGen("dgC2", W_index)];
			Matrix* gC1 = Conv_biases[CharGen("gC1", W_index)];
			Matrix* gC2 = Conv_biases[CharGen("gC2", W_index)];

			/*Updating vdg1, vdg2, sdg1, sdg2*/
			//vdg1 = (vdg1 * (beta1 * momentum)) + (dg1 * (1 - beta1 * momentum));
			Matptr = vdgC1;
			temp1 = vdgC1->mul(beta1 * momentum);
			temp2 = dgC1->mul(1 - beta1 * momentum);
			vdgC1 = temp1->add(temp2);
			delete Matptr;
			delete temp1;
			delete temp2;

			ADAM_dbC.replace(CharGen("vdg1C", W_index), vdgC1);

			//vdg2 = (vdg2 * (beta1 * momentum)) + (dg2 * (1 - beta1 * momentum));
			Matptr = vdgC2;
			temp1 = vdgC2->mul(beta1 * momentum);
			temp2 = dgC2->mul(1 - beta1 * momentum);
			vdgC2 = temp1->add(temp2);
			delete Matptr;
			delete temp1;
			delete temp2;

			ADAM_dbC.replace(CharGen("vdg2C", W_index), vdgC2);

			//sdg1 = (sdg1 * beta2) + (dg1.square() * (1 - beta2));
			Matptr = sdgC1;
			temp1 = sdgC1->mul(beta2);
			temp2 = dgC1->SQUARE();
			temp3 = temp2->mul(1 - beta2);
			sdgC1 = temp1->add(temp3);
			delete Matptr;
			delete temp1;
			delete temp2;
			delete temp3;

			ADAM_dbC.replace(CharGen("sdg1C", W_index), sdgC1);

			//sdg2 = (sdg2 * beta2) + (dg2.square() * (1 - beta2));
			Matptr = sdgC2;
			temp1 = sdgC2->mul(beta2);
			temp2 = dgC2->SQUARE();
			temp3 = temp2->mul(1 - beta2);
			sdgC2 = temp1->add(temp3);
			delete Matptr;
			delete temp1;
			delete temp2;
			delete temp3;

			ADAM_dbC.replace(CharGen("sdg2C", W_index), sdgC2);

			/*Correcting first iterations*/
			Matrix* vdgC1_corr = vdgC1->div(1 - pow(beta1, iteration + 1));
			Matrix* vdgC2_corr = vdgC2->div(1 - pow(beta1, iteration + 1));
			Matrix* sdgC1_corr = sdgC1->div(1 - pow(beta2, iteration + 1));
			Matrix* sdgC2_corr = sdgC2->div(1 - pow(beta2, iteration + 1));


			/*Updating parameters*/

			//Matrix temp1 = vdg1_corr / (sdg1_corr.Sqrt() + epsilon);
			//Matrix g1u = g1 - temp1 * alpha;
			temp1 = sdgC1_corr->SQRT();
			temp2 = temp1->add(epsilon);
			temp3 = vdgC1_corr->div(temp2);
			temp4 = temp3->mul(alpha);
			Matrix* gC1u = gC1->sub(temp4);
			delete gC1;
			delete vdgC1_corr;
			delete sdgC1_corr;
			delete temp1;
			delete temp2;
			delete temp3;
			delete temp4;

			Conv_biases.replace(CharGen("gC1", W_index), gC1u);

			//Matrix temp = vdg2_corr / (sdg2_corr.Sqrt() + epsilon);
			//Matrix g2u = g2 - temp * alpha;
			temp1 = sdgC2_corr->SQRT();
			temp2 = temp1->add(epsilon);
			temp3 = vdgC2_corr->div(temp2);
			temp4 = temp3->mul(alpha);
			Matrix* gC2u = gC2->sub(temp4);
			delete gC2;
			delete vdgC2_corr;
			delete sdgC2_corr;
			delete temp1;
			delete temp2;
			delete temp3;
			delete temp4;

			Conv_biases.replace(CharGen("gC2", W_index), gC2u);

			/*Erasing dgamma1, dgamma2*/
			Conv_dbiases[ThreadNum].DeleteThenErase(CharGen("dgC1", W_index));
			Conv_dbiases[ThreadNum].DeleteThenErase(CharGen("dgC2", W_index));
		}
	}
    /*END OF ADAM OPTIMIZER*/
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::ConvBackwardOptimized(int stride, int A_index, ActivationType activation, int ThreadNum)
{
	bool batchNorm = Arg->batchNorm;
	string str;
	if (A_index == 1)
		str = "AC";
	else
		str = "ACP";

	VectVolume ACprev = Conv_Cache[ThreadNum][CharGen(str, A_index - 1)];
	if (batchNorm)
		str = "ZCn";
	else
		str = "ZC";
	VectVolume ZC = Conv_Cache[ThreadNum][CharGen(str, A_index)];
	VectVolume dAC = Conv_Grades[ThreadNum][CharGen("dAC", A_index)];
	VectVolume WC = Conv_Weights[CharGen("WC", A_index)];
	VectVolume dZC = Calc_dZC(ZC, dAC, activation);
	if (batchNorm)
		Conv_BN_backprop(dZC, A_index, ThreadNum);

	int m = dZC.size();
	int n_C = dZC[0].size();
	int n_H = dZC[0][0]->Rows();
	int n_W = dZC[0][0]->Columns();
	int f = WC[0][0]->Rows();

	VectVolume dACPprev(m, ACprev[0].size(), ACprev[0][0]->Rows(), ACprev[0][0]->Columns());
	VectVolume dWC(WC.size(), WC[0].size(), WC[0][0]->Rows(), WC[0][0]->Columns());

    Matrix* dbC = nullptr;
    if(!batchNorm)
        dbC = new Matrix(n_C, 1);

	Matrix* Matptr1 = nullptr;
	Matrix* Matptr2 = nullptr;
	Matrix* Matptr3 = nullptr;

	for (int i = 0; i < m; i++)
	{
		Volume a_prev = ACprev[i];
		for (int c = 0; c < n_C; c++)
		{
			for (int h = 0; h < n_H; h++)
			{
				for (int w = 0; w < n_W; w++)
				{
					int vert_start = h;
					int vert_end = h + f;
					int horiz_start = w;
					int horiz_end = w + f;
					int dWC_channels = dWC[0].size();

					if (A_index != 1)
						for (int ii = 0; ii < dWC_channels; ii++)
							for (int jj = vert_start, jjW = 0; jj < vert_end; jj++, jjW++)
								for (int kk = horiz_start, kkW = 0; kk < horiz_end; kk++, kkW++)
									dACPprev[i][ii]->access(jj, kk) = dACPprev[i][ii]->access(jj, kk) + WC[c][ii]->access(jjW, kkW) * dZC[i][c]->access(h, w);

					for (int ii = 0; ii < dWC_channels; ii++)
					{
						//dWC[c][ii] = dWC[c][ii] + a_prev[ii]->SubMat(vert_start, horiz_start, vert_end - 1, horiz_end - 1) * dZC[i][c]->access(h, w);
						Matptr1 = a_prev[ii]->SubMat(vert_start, horiz_start, vert_end - 1, horiz_end - 1);
						Matptr2 = Matptr1->mul(dZC[i][c]->access(h, w));
						Matptr3 = dWC[c][ii];
						dWC[c][ii] = Matptr3->add(Matptr2);
						delete Matptr1;
						delete Matptr2;
						delete Matptr3;
					}

					if (!batchNorm)
					{
						//dbC = dbC + dZC[i][c].access(h, w);
						Matptr1 = dbC;
						dbC = Matptr1->add(dZC[i][c]->access(h, w));
						delete Matptr1;
					}
				}
			}
		}
	}

	for (int i = 0; i < dWC.size(); i++)
		for (int j = 0; j < dWC[0].size(); j++)
		{
			//dWC[i][j] = dWC[i][j] / m;
			Matptr1 = dWC[i][j];
			dWC[i][j] = Matptr1->div(m);
			delete Matptr1;
		}


	if(!batchNorm)
    {
        //dbC = dbC / m;
        Matptr1 = dbC;
        dbC = Matptr1->div(m);
        delete Matptr1;
    }

	dZC.DELETE();

	if (A_index != 1)
		Conv_Grades[ThreadNum].put(CharGen("dACP", A_index - 1), dACPprev);
    else
        dACPprev.DELETE();


	Conv_Grades[ThreadNum].put(CharGen("dWC", A_index), dWC);
	if(!batchNorm)
        Conv_dbiases[ThreadNum].put(CharGen("dbC", A_index), dbC);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
VectVolume Calc_dZC(VectVolume ZC, VectVolume dAC, ActivationType activation)
{
	int m = dAC.size();
	int numOfVolumes = dAC[0].size();
	VectVolume dZC(m, numOfVolumes);
	Matrix* dactiv_z = nullptr;

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < numOfVolumes; j++)
		{
			dactiv_z = dactiv(ZC[i][j], activation);

			dZC[i][j] = dAC[i][j]->mul(dactiv_z);

			delete dactiv_z;
		}
	}
	return dZC;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::ConvBackward(int stride, int A_index, ActivationType activation, int ThreadNum)
{
    /*ARGUMENT LIST*/
    bool batchNorm = Arg->batchNorm;
    /*END OF ARGUMENT LIST*/

    /*NECESSARY VARIABLES*/
    string str;
    if(A_index == 1)
        str = "AC";
    else
        str = "ACP";

    VectVolume ACprev = Conv_Cache[ThreadNum][CharGen(str, A_index - 1)];

    if (batchNorm)
		str = "ZCn";
	else
		str = "ZC";
    VectVolume ZC = Conv_Cache[ThreadNum][CharGen(str, A_index)];

    VectVolume dAC = Conv_Grades[ThreadNum][CharGen("dAC", A_index)];
    VectVolume filters = Conv_Weights[CharGen("WC", A_index)];
    VectVolume dZC = Calc_dZC(ZC, dAC, activation);
    /*END OF NECESSARY VARIABLES*/

    if(batchNorm)
		Conv_BN_backprop(dZC, A_index, ThreadNum);

    clock_t start = clock();
    VectVolume dWC = FilterGrades(ACprev, dZC);
    //cout<<"FilterGrades Time = "<<clock() - start<<endl;

    Matrix* dbC = nullptr;
    if(!batchNorm)
        dbC = biasGrades(dZC);

	if(A_index != 1)
    {
        clock_t start = clock();
        VectVolume dACPprev = FullConvolution(dZC, filters);
       // cout<<"FullConvolution Time = "<<clock() - start<<endl;
        Conv_Grades[ThreadNum].put(CharGen("dACP", A_index - 1), dACPprev);
    }

    Matrix* Matptr = nullptr;
    int m = dZC.size();

    for (int i = 0; i < dWC.size(); i++)
    {
        for (int j = 0; j < dWC[0].size(); j++)
		{
			//dWC[i][j] = dWC[i][j] / m;
			Matptr = dWC[i][j];
			dWC[i][j] = Matptr->div(m);
			delete Matptr;
		}
    }
    Conv_Grades[ThreadNum].put(CharGen("dWC", A_index), dWC);

	if(!batchNorm)
    {
        //dbC = dbC / m;
        Matptr = dbC;
        dbC = Matptr->div(m);
        delete Matptr;
        Conv_dbiases[ThreadNum].put(CharGen("dbC", A_index), dbC);
    }

	dZC.DELETE();
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
VectVolume FullConvolution(VectVolume dZC, VectVolume filters)
{
    //Rotate filters 180 degrees and rearrange them
	VectVolume Temp = RotateAllVolumes(filters);
    VectVolume RotatedFilters = RearrangeFilters(Temp);
    Temp.DELETE();

    //Determine the amount of padding required for dZC (p = filter size - 1)
	int p = RotatedFilters[0][0]->Rows() - 1;
	VectVolume Padded_dZC = PadAllVolumes(dZC, p, 0);

	//cout<<"fullConv "<<endl;
    VectVolume A = convolve_Threaded(Padded_dZC, RotatedFilters, 1);

    RotatedFilters.DELETE();
    Padded_dZC.DELETE();

	return A;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
VectVolume FilterGrades(VectVolume ACprev, VectVolume dZC)
{
    //Modify dimensions
    VectVolume dZC_Modified = RearrangeFilters(dZC);
    VectVolume ACprev_Modified = RearrangeFilters(ACprev);

    //cout<<"filterGrades"<<endl;
    VectVolume Temp = convolve(ACprev_Modified, dZC_Modified, 1);
    VectVolume dWC = RearrangeFilters(Temp);

    dZC_Modified.DELETE();
    ACprev_Modified.DELETE();
    Temp.DELETE();

	return dWC;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* biasGrades(VectVolume dZC)
{
    int m = dZC.size();
    int nc = dZC[0].size();
    Matrix* db = new Matrix(nc, 1, 0);
    for (int i = 0; i < nc; i++)
	{
	    int temp = 0;
	    for(int k = 0; k < m; k++)
            temp += dZC[k][i]->sumall();
        db->access(i, 0) = temp;
	}
	return db;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
VectVolume RearrangeFilters(VectVolume filter)
{
    VectVolume Result(filter[0].size(), filter.size());
	for (int i = 0; i < filter[0].size(); i++)
		for (int j = 0; j < filter.size(); j++)
        {
            Result[i][j] = new Matrix(1,1);
            *(Result[i][j]) = *(filter[j][i]);
        }

	return Result;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
VectVolume PadAllVolumes(VectVolume Original, int p, int value)
{
    VectVolume Result(Original.size(), Original[0].size());
	for (int i = 0; i < Original.size(); i++)
        for (int j = 0; j < Original[i].size(); j++)
            Result[i][j] = pad(Original[i][j], p, value);
    return Result;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
VectVolume RotateAllVolumes(VectVolume filters)
{
	VectVolume RotatedFilters(filters.size(), filters[0].size());
	for (int i = 0; i<filters.size(); i++)
        for(int j = 0; j < filters[0].size(); j++)
            RotatedFilters[i][j] = (filters[i][j])->ROT180();

    return RotatedFilters;
}
