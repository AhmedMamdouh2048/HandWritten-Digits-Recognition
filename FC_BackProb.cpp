#include "NeuralNetwork.h"
#include "NN_Tools.h"
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::FC_CalGrads(Matrix* cur_Y, Matrix* Y_hat, int ThreadNum)
{
    /*ARGUMENT LIST*/
    TypeOfNet TN  = Arg->NetType;
    float lambda = Arg->regularizationParameter;
    bool batchNorm = Arg->batchNorm;
    bool dropout = Arg->dropout;
    bool dropConnect = Arg->dropConnect;
    float* keep_prob = Arg->keep_prob;
    layer* layers = Arg->layers;
    int L = Arg->numOfLayers;
    float m = cur_Y->Columns();
    ErrorType ErrType = Arg->ErrType;
    /*END OF ARGUMENT LIST*/

    //////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////

	/*INITIALIZATION OF PARAMETERS FOR BATCHNORM*/
	Matrix** dg1 = new Matrix*[L - 1];   //mean of z for each layer
	Matrix** dg2 = new Matrix*[L - 1];   //standard deviation of z for each layer
	Matrix* g1 = nullptr;
	Matrix* var = nullptr;
	Matrix* dbLast = nullptr;
	Matrix* dbl = nullptr;
	float eps = 1e-7;
	/*END OF INITIALIZATION OF PARAMETERS FOR BATCHNORM*/

	//////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////

	/*TEMPORARY POINTERS*/
	Matrix* MatPtr1 = nullptr;
	Matrix* MatPtr2 = nullptr;
	Matrix* MatPtr3 = nullptr;
	Matrix* MatPtr4 = nullptr;
	/*END OF TEMPORARY POINTERS*/

	//////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////

	/*OUTPUT LAYER*/
	Matrix* dzLast = nullptr;
	Matrix* zLast = nullptr;
	if (batchNorm)
		zLast = FC_Cache[ThreadNum][CharGen("zn", L - 1)];
	else
		zLast = FC_Cache[ThreadNum][CharGen("z", L - 1)];

	if (ErrType == SQAURE_ERROR) //not compatible with dropout
	{
		//Matrix dALast = (Y_hat - Y) / m;
		Matrix* dALast = Y_hat->sub(cur_Y);
		MatPtr1 = dALast;
		dALast = dALast->div(m);
		delete MatPtr1;

		//dzLast = dALast * dactiv(zLast);
		ActivationType activation = layers[L - 1].activation;
		MatPtr1 = dactiv(zLast, activation);
		dzLast = dALast->mul(MatPtr1);
		delete MatPtr1;

	}
	else if (ErrType == CROSS_ENTROPY)
	{
		//dzLast = Y_hat - Y;
		dzLast = Y_hat->sub(cur_Y);
	}

	if (batchNorm)
	{
		//visit https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
		//visit https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
		g1 = FC_Parameters[CharGen("g1", L - 1)];
		var = FC_Cache[ThreadNum][CharGen("var", L - 1)];
		Matrix* z_telda = FC_Cache[ThreadNum][CharGen("zt", L - 1)];

		/*getting dgamma and dbeta*/
		/*dzlast here means dzlast_new after normalizing*/

		//*dg1[L - 2] = (dzLast*z_telda).sum("column")
		MatPtr1 = dzLast->mul(z_telda);
		dg1[L - 2] = MatPtr1->SUM("column");
		delete MatPtr1;

		//*dg2[L - 2] = dzLast.sum("column")
		dg2[L - 2] = dzLast->SUM("column");

		//////////////////////////////////////////////////////
		/*getting dz_telda*/
		//dz_telda = dzLast * (g1)
		Matrix* dz_telda = dzLast->mul(g1);

        //////////////////////////////////////////////////////
		/*getting dvariance*/
		Matrix* zmeu = FC_Cache[ThreadNum][CharGen("zm", L - 1)];

		//divar = (dz_telda*zmeu).sum("column")
		MatPtr1 = dz_telda->mul(zmeu);
		Matrix* divar = MatPtr1->SUM("column");
		delete MatPtr1;

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
		MatPtr1 = dz_telda->mul(-1);
		Matrix* dmeu1 = MatPtr1->div(t);
		delete MatPtr1;

		//dmeu1 = dmeu1.sum("column");
		MatPtr1 = dmeu1;
		dmeu1 = dmeu1->SUM("column");
		delete MatPtr1;

		//dmeu2 = (zmeu*-2)*dvar;
		MatPtr1 = zmeu->mul(-2);
		Matrix* dmeu2 = MatPtr1->mul(dvar);
		delete MatPtr1;

		//dmeu2 = (dmeu2.sum("column")) / m;
		MatPtr1 = dmeu2->SUM("column");
		MatPtr2 = dmeu2;
		dmeu2 = MatPtr1->div(m);
		delete MatPtr2;
		delete MatPtr1;

		//dmeu = dmeu1 + dmeu2;
		Matrix* dmeu = dmeu1->add(dmeu2);

		//////////////////////////////////////////////////////
		/*getting dzlast (dout) for the incoming layer*/
		/*
		This matrix (dzlast) contains the gradient of the loss function with respect to the input of the BatchNorm-Layer.
		This dzlast is the gradient of zlast=W.A[L-1]+b (b is neglected)
		This gradient dzlast is also what we give as input (dout) to the backwardpass of the next layer..this happens through dAprevLast
		As for this layer we receive dout from the layer above.
		*/

		//dzLast = dz_telda / t;
		MatPtr1 = dzLast;
		dzLast = dz_telda->div(t);
		delete MatPtr1;

		//dzLast = dzLast + (zmeu*dvar)*(2 / m);
		MatPtr1 = dzLast;
		MatPtr2 = zmeu->mul(dvar);
		MatPtr3 = MatPtr2->mul(2 / m);
		dzLast = dzLast->add(MatPtr3);
		delete MatPtr1;
		delete MatPtr2;
		delete MatPtr3;

		//dzLast = dzLast + dmeu / m;
		MatPtr1 = dzLast;
		MatPtr2 = dmeu->div(m);
		dzLast = dzLast->add(MatPtr2);
		delete MatPtr1;
		delete MatPtr2;

		FC_Grades[ThreadNum].put(CharGen("dg1", L - 1), dg1[L - 2]);
		FC_Grades[ThreadNum].put(CharGen("dg2", L - 1), dg2[L - 2]);

		delete dz_telda;
		delete divar;
		delete dsqrtvar;
		delete t;
		delete dvar;
		delete dmeu1;
		delete dmeu2;
		delete dmeu;
	}

	Matrix* AprevLast = FC_Cache[ThreadNum][CharGen("A", L - 2)];

	//dWLast = dzLast.dot(AprevLast.transpose()) / m;
	MatPtr1 = AprevLast->TRANSPOSE();
    MatPtr2 = DOT(dzLast, MatPtr1);
	Matrix* dWLast = MatPtr2->div(m);
	delete MatPtr1;
	delete MatPtr2;

	if (!batchNorm)
	{
		//dbLast = dzLast.sum("column") / m;
		MatPtr1 = dzLast->SUM("column");
		dbLast = MatPtr1->div(m);
		delete MatPtr1;
	}

	// dAprevLast = WLast.transpose().dot(dzLast)
	Matrix* WLast = FC_Parameters[CharGen("W", L - 1)];
	Matrix* WLast_trans = WLast->TRANSPOSE();
    Matrix* dAprevLast = DOT(WLast_trans, dzLast);
	delete WLast_trans;
	delete dzLast;

	if (dropout)
	{
		//dAprevLast = dAprevLast * (*D[L - 2]);
		MatPtr1 = dAprevLast;
		dAprevLast = dAprevLast->mul(D[ThreadNum][L - 2]);
		delete MatPtr1;

		//dAprevLast = dAprevLast / keep_prob[L - 2];
		MatPtr1 = dAprevLast;
		dAprevLast = dAprevLast->div(keep_prob[L - 2]);
		delete MatPtr1;
	}

	if (lambda != 0)
	{
		//dWLast = dWLast + WLast * (lambda / m);
		MatPtr1 = dWLast;
		MatPtr2 = WLast->mul(lambda / m);
		dWLast = dWLast->add(MatPtr2);
		delete MatPtr1;
		delete MatPtr2;
	}

	FC_Grades[ThreadNum].put(CharGen("dW", L - 1), dWLast);

	if (!batchNorm)
		FC_Grades[ThreadNum].put(CharGen("db", L - 1), dbLast);

	FC_Grades[ThreadNum].put(CharGen("dA", L - 2), dAprevLast);
	/*END OF OUTPUT LAYER*/

    //////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////

	/*HIDDEN LAYERS*/
	Matrix* dAl_1 = nullptr;
	for (int i = L - 2; i > 0; i--)
	{
		Matrix* dAl = FC_Grades[ThreadNum][CharGen("dA", i)];
		FC_Grades[ThreadNum].erase(CharGen("dA", i));
		Matrix* zl;
		Matrix* dzl;

		if (batchNorm)
			zl = FC_Cache[ThreadNum][CharGen("zn", i)];
		else
			zl = FC_Cache[ThreadNum][CharGen("z", i)];

		//dzl = dAl * dactiv(zl);
		ActivationType activation = layers[i].activation;
		MatPtr1 = dactiv(zl, activation);
		dzl = dAl->mul(MatPtr1);
		delete MatPtr1;

		if (batchNorm)
		{
			g1 = FC_Parameters[CharGen("g1", i)];
			var = FC_Cache[ThreadNum][CharGen("var", i)];
			Matrix* z_telda = FC_Cache[ThreadNum][CharGen("zt", i)];

			/*getting dgamma and dbeta*/
			/*dz here means dznew after normalizing*/

			//*dg1[i - 1] = (dzl*z_telda).sum("column");
			MatPtr1 = dzl->mul(z_telda);
			dg1[i - 1] = MatPtr1->SUM("column");
			delete MatPtr1;

			//dg2[i - 1] = dzl.sum("column");
			dg2[i - 1] = dzl->SUM("column");

			//////////////////////////////////////////////////////
			/*getting dz_telda*/
			//dz_telda = dzl * (g1);
			Matrix* dz_telda = dzl->mul(g1);

			//////////////////////////////////////////////////////
			/*getting dvariance*/
			Matrix* zmeu = FC_Cache[ThreadNum][CharGen("zm", i)];

			//divar = (dz_telda*zmeu).sum("column");
			MatPtr1 = dz_telda->mul(zmeu);
			Matrix* divar = MatPtr1->SUM("column");
			delete MatPtr1;

			//dsqrtvar = divar / (var + eps);
			MatPtr1 = var->add(eps);
			Matrix* dsqrtvar = divar->div(MatPtr1);
			delete MatPtr1;

			//t = (var + eps).Sqrt();
			MatPtr1 = var->add(eps);
			Matrix* t = MatPtr1->SQRT();
			delete MatPtr1;

			//dvar = (dsqrtvar*-0.5) / t;
			MatPtr1 = dsqrtvar->mul(-0.5);
			Matrix* dvar = MatPtr1->div(t);
			delete MatPtr1;

			//////////////////////////////////////////////////////
			/*getting dmeu*/
			//Matrix dmeu1 = (dz_telda*-1) / t;
			MatPtr1 = dz_telda->mul(-1);
			Matrix* dmeu1 = MatPtr1->div(t);
			delete MatPtr1;

			//dmeu1 = dmeu1.sum("column");
			MatPtr1 = dmeu1;
			dmeu1 = dmeu1->SUM("column");
			delete MatPtr1;

			//dmeu2 = (zmeu*-2)*dvar;
			MatPtr1 = zmeu->mul(-2);
			Matrix* dmeu2 = MatPtr1->mul(dvar);
			delete MatPtr1;

			//dmeu2 = (dmeu2.sum("column")) / m;
			MatPtr1 = dmeu2->SUM("column");
			MatPtr2 = dmeu2;
			dmeu2 = MatPtr1->div(m);
			delete MatPtr2;
			delete MatPtr1;

			//dmeu = dmeu1 + dmeu2;
			Matrix* dmeu = dmeu1->add(dmeu2);

			//////////////////////////////////////////////////////
			/*getting dz (dout for the incoming layer)*/
			/*
			This matrix (dz) contains the gradient of the loss function with respect to the input of the BatchNorm-Layer.
			This dz is the gradient of z=W.A[l-1]+b (b is neglected)
			This gradient dz is also what we give as input (dout) to the backwardpass of the next layer..this happens through dAl_1
			As for this layer we receive dout from the layer above.
			*/

			//dzl = dz_telda / t;
			MatPtr1 = dzl;
			dzl = dz_telda->div(t);
			delete MatPtr1;

			//dzl = dzl + (zmeu*dvar)*(2 / m);
			MatPtr1 = dzl;
			MatPtr2 = zmeu->mul(dvar);
			MatPtr3 = MatPtr2->mul(2 / m);
			dzl = dzl->add(MatPtr3);
			delete MatPtr1;
			delete MatPtr2;
			delete MatPtr3;

			//dzl = dzl + dmeu / m;
			MatPtr1 = dzl;
			MatPtr2 = dmeu->div(m);
			dzl = dzl->add(MatPtr2);
			delete MatPtr1;
			delete MatPtr2;

			FC_Grades[ThreadNum].put(CharGen("dg1", i), dg1[i - 1]);
			FC_Grades[ThreadNum].put(CharGen("dg2", i), dg2[i - 1]);

			delete dz_telda;
			delete divar;
			delete dsqrtvar;
			delete t;
			delete dvar;
			delete dmeu1;
			delete dmeu2;
			delete dmeu;
		}
		Matrix* Al_1 = FC_Cache[ThreadNum][CharGen("A", i - 1)];

		Matrix* dWl = nullptr;
		if (dropConnect)
        {
            //Matrix* dWl = summation of[{dzl.dot(Al_1.transpose()) for every one example} * {Mask of that example}] / m;
            MatPtr1 = Al_1->TRANSPOSE();
            for(int ii = 0; ii < MatPtr1->Rows(); ii++)
            {
                MatPtr2 = dzl->SubMat(0,ii,-1,ii);
                MatPtr3 = MatPtr1->SubMat(ii,0,ii,-1);
                MatPtr4 = DOT(MatPtr2, MatPtr3);
                delete MatPtr2;
                delete MatPtr3;
                MatPtr2 = MatPtr4->MultBool(D2[ThreadNum][i-1][ii]);
                delete MatPtr4;
                if(ii == 0)
                    dWl = MatPtr2;
                else
                {
                    MatPtr3 = dWl->add(MatPtr2);
                    delete MatPtr2;
                    delete dWl;
                    dWl = MatPtr3;
                }
            }
            delete MatPtr1;

            MatPtr1 = dWl;
            dWl = dWl->div(m * keep_prob[i]);
            delete MatPtr1;
        }
        else
        {
            //Matrix* dWl = dzl.dot(Al_1.transpose()) / m;
            MatPtr1 = Al_1->TRANSPOSE();
            MatPtr2 = DOT(dzl, MatPtr1);
            dWl = MatPtr2->div(m);
            delete MatPtr1;
            delete MatPtr2;
        }

		if (!batchNorm)
		{
			//dbl = dzl->sum("column") / m;
			MatPtr1 = dzl->SUM("column");
			dbl = MatPtr1->div(m);
			delete MatPtr1;
		}

		Matrix* Wl = FC_Parameters[CharGen("W", i)];
		if (i != 1 || TN != FC)
		{
			Matrix* Wl_trans = Wl->TRANSPOSE();
		    dAl_1 = DOT(Wl_trans, dzl);
			delete Wl_trans;
		}

		delete dzl;
		delete dAl;

		if (dropout && i != 1)
		{
			//dAl_1 = dAl_1 * (*(D[i - 1]));
			MatPtr1 = dAl_1;
			dAl_1 = dAl_1->mul(D[ThreadNum][i - 1]);
			delete MatPtr1;

			//dAl_1 = dAl_1 / keep_prob[i - 1];
			MatPtr1 = dAl_1;
			dAl_1 = dAl_1->div(keep_prob[i - 1]);
			delete MatPtr1;
		}

		if (lambda != 0)
		{
			//dWl = dWl + Wl * (lambda / m);
			MatPtr1 = dWl;
			MatPtr2 = Wl->mul(lambda / m);
			dWl = dWl->add(MatPtr2);
			delete MatPtr1;
			delete MatPtr2;
		}

		//if(i == 2)
           // dWl->print();
		FC_Grades[ThreadNum].put(CharGen("dW", i), dWl);

		if (!batchNorm)
			FC_Grades[ThreadNum].put(CharGen("db", i), dbl);

		if (i != 1 || TN!=FC)
			FC_Grades[ThreadNum].put(CharGen("dA", i - 1), dAl_1);
	}

	if (dropout)
	{
		for (int j = 1; j < L - 1; j++)
			delete D[ThreadNum][j];
	}

	if (dropConnect)
    {
        for(int j = 0; j < L - 2; j++)
        {
            for(int k = 0; k < m; k++)
            {
                delete D2[ThreadNum][j][k];
            }
            delete D2[ThreadNum][j];
        }
    }
	/*END OF HIDDEN LAYERS*/
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::FC_UpdateParameters(int iteration, int ThreadNum)
{
    /*ARGUMENT LIST*/
    int L = Arg->numOfLayers;
	bool batchNorm = Arg->batchNorm;
	float alpha = Arg->learingRate;
	Optimizer optimizer = Arg->optimizer;
    /*END OF ARGUMENT LIST*/

    /*TEMPORARY POINTERS*/
	Matrix* Matptr = nullptr;
	Matrix* temp1 = nullptr;
	Matrix* temp2 = nullptr;
	Matrix* temp3 = nullptr;
	Matrix* temp4 = nullptr;
	/*END OF TEMPORARY POINTERS*/

	/*START OF GRADIENT DESCENT OPTIMIZER*/
	if (optimizer == GRADIENT_DESCENT)
	{
		for (int i = 0; i < L - 1; i++)
		{
			if (batchNorm)
			{
			    //g1 = g1 - dg1 * alpha;
				Matrix* g1 = FC_Parameters[CharGen("g1", i + 1)];
				Matrix* dg1 = FC_Grades[ThreadNum][CharGen("dg1", i + 1)];
				Matptr = g1;
				temp1 = dg1->mul(alpha);
				g1 = g1->sub(temp1);
				delete Matptr;
				delete temp1;

				FC_Parameters.replace(CharGen("g1", i + 1), g1);
				FC_Grades[ThreadNum].DeleteThenErase(CharGen("dg1", i + 1));

				//g2 = g2 - dg2 * alpha;
				Matrix* g2 = FC_Parameters[CharGen("g2", i + 1)];
				Matrix* dg2 = FC_Grades[ThreadNum][CharGen("dg2", i + 1)];
				Matptr = g2;
				temp1 = dg2->mul(alpha);
				g2 = g2->sub(temp1);
				delete Matptr;
				delete temp1;

				FC_Parameters.replace(CharGen("g2", i + 1), g2);
				FC_Grades[ThreadNum].DeleteThenErase(CharGen("dg2", i + 1));
			}

			//Wul=W-alpha*dW
			Matrix* W = FC_Parameters[CharGen("W", i + 1)];
			Matrix* dW = FC_Grades[ThreadNum][CharGen("dW", i + 1)];
			temp1 = dW->mul(alpha);
			Matrix* Wul = W->sub(temp1);
			delete W;
			delete temp1;

			FC_Parameters.replace(CharGen("W", i + 1), Wul);
			FC_Grades[ThreadNum].DeleteThenErase(CharGen("dW", i + 1));

			if (!batchNorm)
			{
				//bul=b-alpha*db
				Matrix* b = FC_Parameters[CharGen("b", i + 1)];
				Matrix* db = FC_Grades[ThreadNum][CharGen("db", i + 1)];
				temp1 = db->mul(alpha);
				Matrix* bul = b->sub(temp1);
				delete b;
				delete temp1;

				FC_Parameters.replace(CharGen("b", i + 1), bul);
				FC_Grades[ThreadNum].DeleteThenErase(CharGen("db", i + 1));
			}
		}
	}
	/*END OF GRADIENT DESCENT OPTIMIZER*/

	/*START OF ADAM OPTIMIZER*/
	else if (optimizer == ADAM)
	{
		float beta1 = 0.9;
		float beta2 = 0.999;
		float epsilon = 1e-8;
		Matrix* b = nullptr;
		Matrix* db = nullptr;
		Matrix* Vdb = nullptr;
		Matrix* Sdb = nullptr;
		Matrix* Sdb_corr = nullptr;
		Matrix* Vdb_corr = nullptr;

		for (int i = 0; i < L - 1; i++)
		{
			/*Getting variables from dictionaries*/
			Matrix* Vdw = FC_ADAM[CharGen("Vdw", i + 1)];
			Matrix* Sdw = FC_ADAM[CharGen("Sdw", i + 1)];
			if (!batchNorm)
			{
				Vdb = FC_ADAM[CharGen("Vdb", i + 1)];
				Sdb = FC_ADAM[CharGen("Sdb", i + 1)];
			}

			Matrix* dW = FC_Grades[ThreadNum][CharGen("dW", i + 1)];
			Matrix* W = FC_Parameters[CharGen("W", i + 1)];
			if (!batchNorm)
			{
				db = FC_Grades[ThreadNum][CharGen("db", i + 1)];
				b = FC_Parameters[CharGen("b", i + 1)];
			}

			/*Updating Vdw, Vdb, Sdw, Sdb*/
			//Vdw = (Vdw * (beta1 * momentum)) + (dW * (1 - beta1 * momentum));
			Matptr = Vdw;
			temp1 = Vdw->mul(beta1 * momentum);
			temp2 = dW->mul(1 - beta1 * momentum);
			Vdw = temp1->add(temp2);
			delete Matptr;
			delete temp1;
			delete temp2;

			FC_ADAM.replace(CharGen("Vdw", i + 1), Vdw);

			if (!batchNorm)
			{
				//Vdb = (Vdb * (beta1 * momentum)) + (db * (1 - beta1 * momentum));
				Matptr = Vdb;
				temp1 = Vdb->mul(beta1 * momentum);
				temp2 = db->mul(1 - beta1 * momentum);
				Vdb = temp1->add(temp2);
				delete Matptr;
                delete temp1;
                delete temp2;
				FC_ADAM.replace(CharGen("Vdb", i + 1), Vdb);
			}

			//Sdw = (Sdw * beta2) + (dW.square() * (1 - beta2));
			Matptr = Sdw;
			temp1 = Sdw->mul(beta2);
			temp2 = dW->SQUARE();
			temp3 = temp2->mul(1 - beta2);
			Sdw = temp1->add(temp3);
			delete Matptr;
			delete temp1;
			delete temp2;
			delete temp3;

			FC_ADAM.replace(CharGen("Sdw", i + 1), Sdw);

			if (!batchNorm)
			{
				//Sdb = (Sdb * beta2) + (db.square() * (1 - beta2));
				Matptr = Sdb;
				temp1 = Sdb->mul(beta2);
				temp2 = db->SQUARE();
				temp3 = temp2->mul(1 - beta2);
				Sdb = temp1->add(temp3);
				delete Matptr;
                delete temp1;
                delete temp2;
                delete temp3;

				FC_ADAM.replace(CharGen("Sdb", i + 1), Sdb);
			}

			/*Correcting first iterations*/
			Matrix* Vdw_corr = Vdw->div(1 - pow(beta1, iteration + 1));
			Matrix* Sdw_corr = Sdw->div(1 - pow(beta2, iteration + 1));
			if (!batchNorm)
			{
				Sdb_corr = Sdb->div(1 - pow(beta2, iteration + 1));
				Vdb_corr = Vdb->div(1 - pow(beta1, iteration + 1));
			}

			/*Updating parameters*/

			//Matrix temp = Vdw_corr / (Sdw_corr.Sqrt() + epsilon);
			//Matrix Wu = W - temp * alpha;
			temp1 = Sdw_corr->SQRT();
			temp2 = temp1->add(epsilon);
			temp3 = Vdw_corr->div(temp2);
			temp4 = temp3->mul(alpha);
			Matrix* Wu = W->sub(temp4);
			delete Vdw_corr;
			delete Sdw_corr;
			delete temp1;
			delete temp2;
			delete temp3;
			delete temp4;

			FC_Parameters.DeleteThenReplace(CharGen("W", i + 1), Wu);

			if (!batchNorm)
			{
				//Matrix temp = Vdb_corr / (Sdb_corr.Sqrt() + epsilon);
				//Matrix bu = b - temp * alpha;
				temp1 = Sdb_corr->SQRT();
				temp2 = temp1->add(epsilon);
				temp3 = Vdb_corr->div(temp2);
				temp4 = temp3->mul(alpha);
				Matrix* bu = b->sub(temp4);
				delete Vdb_corr;
                delete Sdb_corr;
                delete temp1;
                delete temp2;
                delete temp3;
                delete temp4;

				FC_Parameters.DeleteThenReplace(CharGen("b", i + 1), bu);
			}

			/*Erasing dW, db*/
			FC_Grades[ThreadNum].DeleteThenErase(CharGen("dW", i + 1));
			if (!batchNorm)
				FC_Grades[ThreadNum].DeleteThenErase(CharGen("db", i + 1));

			if (batchNorm)
			{
				/*Getting variables from dictionaries*/
				Matrix* vdg1 = FC_ADAM[CharGen("vg1", i + 1)];
				Matrix* sdg1 = FC_ADAM[CharGen("sg1", i + 1)];
				Matrix* vdg2 = FC_ADAM[CharGen("vg2", i + 1)];
				Matrix* sdg2 = FC_ADAM[CharGen("sg2", i + 1)];


				Matrix* dg1 = FC_Grades[ThreadNum][CharGen("dg1", i + 1)];
				Matrix* dg2 = FC_Grades[ThreadNum][CharGen("dg2", i + 1)];
				Matrix* g1 = FC_Parameters[CharGen("g1", i + 1)];
				Matrix* g2 = FC_Parameters[CharGen("g2", i + 1)];

				/*Updating vdg1, vdg2, sdg1, sdg2*/
				//vdg1 = (vdg1 * (beta1 * momentum)) + (dg1 * (1 - beta1 * momentum));
				Matptr = vdg1;
				temp1 = vdg1->mul(beta1 * momentum);
				temp2 = dg1->mul(1 - beta1 * momentum);
				vdg1 = temp1->add(temp2);
				delete Matptr;
				delete temp1;
				delete temp2;

				FC_ADAM.replace(CharGen("vg1", i + 1), vdg1);

				//vdg2 = (vdg2 * (beta1 * momentum)) + (dg2 * (1 - beta1 * momentum));
				Matptr = vdg2;
				temp1 = vdg2->mul(beta1 * momentum);
				temp2 = dg2->mul(1 - beta1 * momentum);
				vdg2 = temp1->add(temp2);
				delete Matptr;
				delete temp1;
				delete temp2;

				FC_ADAM.replace(CharGen("vg2", i + 1), vdg2);

				//sdg1 = (sdg1 * beta2) + (dg1.square() * (1 - beta2));
				Matptr = sdg1;
				temp1 = sdg1->mul(beta2);
				temp2 = dg1->SQUARE();
				temp3 = temp2->mul(1 - beta2);
				sdg1 = temp1->add(temp3);
				delete Matptr;
				delete temp1;
				delete temp2;
				delete temp3;

				FC_ADAM.replace(CharGen("sg1", i + 1), sdg1);

				//sdg2 = (sdg2 * beta2) + (dg2.square() * (1 - beta2));
				Matptr = sdg2;
				temp1 = sdg2->mul(beta2);
				temp2 = dg2->SQUARE();
				temp3 = temp2->mul(1 - beta2);
				sdg2 = temp1->add(temp3);
				delete Matptr;
				delete temp1;
				delete temp2;
				delete temp3;

				FC_ADAM.replace(CharGen("sg2", i + 1), sdg2);

				/*Correcting first iterations*/
				Matrix* vdg1_corr = vdg1->div(1 - pow(beta1, iteration + 1));
				Matrix* vdg2_corr = vdg2->div(1 - pow(beta1, iteration + 1));
				Matrix* sdg1_corr = sdg1->div(1 - pow(beta2, iteration + 1));
				Matrix* sdg2_corr = sdg2->div(1 - pow(beta2, iteration + 1));


				/*Updating parameters*/

				//Matrix temp1 = vdg1_corr / (sdg1_corr.Sqrt() + epsilon);
				//Matrix g1u = g1 - temp1 * alpha;
				temp1 = sdg1_corr->SQRT();
				temp2 = temp1->add(epsilon);
				temp3 = vdg1_corr->div(temp2);
				temp4 = temp3->mul(alpha);
				Matrix* g1u = g1->sub(temp4);
				delete g1;
				delete vdg1_corr;
				delete sdg1_corr;
				delete temp1;
				delete temp2;
				delete temp3;
				delete temp4;

				FC_Parameters.replace(CharGen("g1", i + 1), g1u);

				//Matrix temp = vdg2_corr / (sdg2_corr.Sqrt() + epsilon);
				//Matrix g2u = g2 - temp * alpha;
				temp1 = sdg2_corr->SQRT();
				temp2 = temp1->add(epsilon);
				temp3 = vdg2_corr->div(temp2);
				temp4 = temp3->mul(alpha);
				Matrix* g2u = g2->sub(temp4);
				delete g2;
				delete vdg2_corr;
				delete sdg2_corr;
				delete temp1;
				delete temp2;
				delete temp3;
				delete temp4;

				FC_Parameters.replace(CharGen("g2", i + 1), g2u);

				/*Erasing dgamma1, dgamma2*/
				FC_Grades[ThreadNum].DeleteThenErase(CharGen("dg1", i + 1));
				FC_Grades[ThreadNum].DeleteThenErase(CharGen("dg2", i + 1));

			}
		}
	}
	/*END OF ADAM OPTIMIZER*/
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
