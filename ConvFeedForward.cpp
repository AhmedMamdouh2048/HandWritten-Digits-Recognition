#include "NeuralNetwork.h"

void NeuralNetwork::convLayer(Mode mode,int stride, int A_index, ActivationType activation, int ThreadNum)
{
	//We will use str to extract Aprev, Aprev is the input data only in the first layer, after that Aprev comes from the pooling layer
	bool batchNorm = Arg->batchNorm;
	string str;
	if (A_index == 1)
		str = "AC";
	else
		str = "ACP";

	VectVolume Aprev = Conv_Cache[ThreadNum][CharGen(str, A_index - 1)];
	VectVolume filters = Conv_Weights[CharGen("WC", A_index)];
	Matrix* b =nullptr;
	if(!batchNorm)
      b= Conv_biases[CharGen("bC", A_index)];

	int m = Aprev.size();
	int numOfFilters = filters.size();

	VectVolume A(m, numOfFilters); //
	VectVolume Z(m, numOfFilters); //

	Matrix* z = nullptr;
	Matrix* a = nullptr;

    //cout<<"convLayer "<<endl;
	VectVolume ConvResult = convolve_Threaded(Aprev, filters, stride);

	for (int i = 0; i<m; i++)
	{
		for (int j = 0; j<numOfFilters; j++)
		{
			if (!batchNorm)
			{
				//add the bias to the result of convolution (*z) = (*z) + b.access(j,0);
				z = ConvResult[i][j]->add(b->access(j, 0));

				//store z, needed later
				Z[i][j] = z;

				//pass the result to the activation a=activation(z)
				a = activ(Z[i][j], activation);

				//a is pointer to the output of convolution, push it into the volume A[i]
				A[i][j] = a;
			}
			else
			{
			    Z[i][j] = ConvResult[i][j];
			}
		}
	}
	Conv_Cache[ThreadNum].put(CharGen("ZC", A_index), Z);
	if (batchNorm)
	{
		Conv_BN_feedforward(mode, A_index, ThreadNum);
		VectVolume ZCnew= Conv_Cache[ThreadNum][CharGen("ZCn", A_index)];
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < numOfFilters; j++)
			{
				//pass the result to the activation a=activation(z)
				a = activ(ZCnew[i][j], activation);
				//a is pointer to the output of convolution, push it into the volume A[i]
				A[i][j] = a;
			}
		}
	}

	Conv_Cache[ThreadNum].put(CharGen("AC", A_index), A);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::poolLayer(int stride, int f, Mode mode, int A_index, int ThreadNum)
{
	VectVolume Aprev = Conv_Cache[ThreadNum][CharGen("AC", A_index)];

	int m = Aprev.size();

	VectVolume A(m, Aprev[0].size());

	for (int i = 0; i<m; i++)
	{
		if (mode == MAX)
			maxPool(Aprev[i], A[i], f, stride);
		else if (mode == AVG)
			avgPool(Aprev[i], A[i], f, stride);
	}
	Conv_Cache[ThreadNum].put(CharGen("ACP", A_index), A);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* to_1D(Volume& X_2D)
{
	Matrix* X_1D = new Matrix(X_2D[0]->Rows() * X_2D[0]->Columns(), X_2D.size());

	for (int k = 0; k < X_2D.size(); k++)
	{
		Matrix* curImg = X_2D[k];
		for (int i = 0; i < curImg->Rows(); i++)
		{
			for (int j = 0; j < curImg->Columns(); j++)
			{
				X_1D->access(i * curImg->Columns() + j, k) = curImg->access(i, j);
			}
		}
	}
	return X_1D;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* to_FC(VectVolume A)
{
    int nh=A[0][0]->Rows();
    int nw=A[0][0]->Columns();
    int nc=A[0].size();
    int m=A.size();
	Matrix* A_1D = new Matrix(nh*nw*nc,m);

	for (int k = 0; k < m; k++)
	{
	    for(int kk=0;kk<nc;kk++)
        {
            for(int i=0;i<nh;i++)
            {
                for(int j=0;j<nw;j++)
                {
                    A_1D->access(j+nh*i+kk*nc,k)=A[k][kk]->access(i,j);
                }
            }
        }
	}
	return A_1D;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Volume to_2D(Matrix* X)
{
	int numOfImgs = X->Columns();
	int dim = sqrt(X->Rows());
	Volume X_2D(numOfImgs);
	for (int k = 0; k < numOfImgs; k++)
	{
		X_2D[k] = new Matrix(dim, dim);
		for(int i=0; i<dim; i++)
			for (int j = 0; j < dim; j++)
			{
				X_2D[k]->access(i, j) = X->access(i*dim + j, k);
			}
	}
	return X_2D;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
VectVolume to_VectorOfVolume(Matrix* A, int nh, int nw, int nc, int m)
{
	VectVolume V(m,nc,nh,nw);
	for (int k = 0; k < m; k++)
	{
		for (int kk = 0; kk<nc; kk++)
		{
			for (int i = 0; i<nh; i++)
			{
				for (int j = 0; j<nw; j++)
				{
					V[k][kk]->access(i, j) = A->access(j + nh * i + kk * nc, k);
				}
			}
		}
	}
	return V;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* pad(Matrix* img, int p, float value)
{
	if (img->Rows() != img->Columns())
		cout << "this is not square matrix" << endl;

	int n = img->Rows();
	int m = n + 2 * p;

	Matrix* newImg = new Matrix(m, m);

	for (int i = 0; i < m; i++)
		for (int j = 0; j < m; j++)
		{
			if (i < (m - n - p) || j<(m - n - p) || i>(p + n - 1) || j >(p + n - 1))
				newImg->access(i, j) = 0;
			else
				newImg->access(i, j) = img->access(i - p, j - p);
		}

	return newImg;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* convolve(Volume& Aprev, Volume& filter, int stride)
{
	int nc = filter.size();
	int f = filter[0]->Rows();
	int nc_prev = Aprev.size();
	int nh_prev = Aprev[0]->Rows();
	int nw_prev = Aprev[0]->Columns();
	int nh = (nh_prev - f) / stride + 1;
	int nw = (nw_prev - f) / stride + 1;
	Matrix* result = new Matrix(nh, nw);
	Matrix* Acc = nullptr;
	Matrix* slice = nullptr;
	Matrix* temp1 = nullptr;
	Matrix* temp2 = nullptr;

	if (nc != nc_prev)
	{
		cout << "dimension err in convolution!" << endl;
	}

	for (int i = 0; i < nh; i++)
	{

		for (int j = 0; j<nw; j++)
		{
			int vert_start = i * stride;
			int vert_end = vert_start + f;
			int horz_start = j * stride;
			int horz_end = horz_start + f;
			Acc = new Matrix(f, f, 0);

			for (int c = 0; c < nc; c++)
			{
				//slice = Aprev[c](vert_start, horz_start, vert_end - 1, horz_end - 1);
				slice = Aprev[c]->SubMat(vert_start, horz_start, vert_end - 1, horz_end - 1);

				//Acc = Acc + slice * filter[c];
				temp1 = slice->mul(filter[c]);
				temp2 = Acc;
				Acc = Acc->add(temp1);

				delete slice;
				delete temp1;
				delete temp2;
			}


			result->access(i, j) = Acc->sumall();
			delete Acc;
		}
	}


	return result;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void maxPool(Volume& Aprev, Volume& A, int f, int stride)
{
	int nc = Aprev.size();
	int nh_prev = Aprev[0]->Rows();
	int nw_prev = Aprev[0]->Columns();
	int nh = (nh_prev - f) / stride + 1;
	int nw = (nw_prev - f) / stride + 1;
	Matrix* slice = nullptr;

	for (int c = 0; c < nc; c++)
	{
		A[c] = new Matrix(nh, nw);
		for (int i = 0; i<nh; i++)
			for (int j = 0; j<nw; j++)
			{
				int vert_start = i * stride;
				int vert_end = vert_start + f;
				int horz_start = j * stride;
				int horz_end = horz_start + f;

				slice = Aprev[c]->SubMat(vert_start, horz_start, vert_end - 1, horz_end - 1);

				A[c]->access(i, j) = slice->MaxElement();

				delete slice;
			}
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void avgPool(Volume& Aprev, Volume& A, int f, int stride)
{
	int nc = Aprev.size();
	int nh_prev = Aprev[0]->Rows();
	int nw_prev = Aprev[0]->Columns();
	int nh = (nh_prev - f) / stride + 1;
	int nw = (nw_prev - f) / stride + 1;
	Matrix* slice = nullptr;

	for (int c = 0; c < nc; c++)
	{
		A[c] = new Matrix(nh, nw);
		for (int i = 0; i<nh; i++)
			for (int j = 0; j<nw; j++)
			{
				int vert_start = i * stride;
				int vert_end = vert_start + f;
				int horz_start = j * stride;
				int horz_end = horz_start + f;

				slice = Aprev[c]->SubMat(vert_start, horz_start, vert_end - 1, horz_end - 1);

				float sum = slice->sumall();

				A[c]->access(i, j) = sum / (slice->Rows() * slice->Columns());

				delete slice;
			}
	}

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* FilterToMatrix(Matrix* filter, int nh, int nw, int s)
{
    int f = filter->Rows();                 // Filter size
    int n = nh * nw;                        // 2D filter row size
    int p = (nh - f + 1) * (nw - f + 1);    // 2D filter column size
    Matrix* Result = new Matrix(p, n, 0);

    int count3 = 0;                         // Counter for the number of shifts from the diagonal
    for(int i = 0; i < p; i++)
    {
        int count1 = f * f;                 // Counter for the number of elements in the filter
        int count2 = 0;                     // Counter for the number of elements in a single row in the filter
        int ii = 0;                         // Row index for the filter
        int jj = 0;                         // Column index for the filter
        for(int j = i + count3; j < n; j++)
        {
            if(i != 0 && j == i + count3 && i % (nh - f + 1) == 0)
            {
                // Shift by f-1 for every down movement of the filter
                j += f - 1;
                count3 += f - 1;
            }
            if(count2 != 0 && count2 % f == 0)
            {
                // Shift by s for every right movement of the filter
                j += nh - f;
                ii++;
                count2 = 0;
            }

            Result->access(i, j) = filter->access(ii, jj);

            count2++;

            // If all elements of the filter are placed in the row
            count1--;
            if(count1 == 0)
                break;

            // If the row of the filter is placed
            jj++;
            if(jj == f)
                jj = 0;
        }

    }
    return Result;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Volume Imgs2Vects(VectVolume Imgs)
{
    int m = Imgs.size();
    int nc = Imgs[0].size();
    int nh = Imgs[0][0]->Rows();
    int nw = Imgs[0][0]->Columns();

    Volume Vects(nc, nh * nw, m);
    int countx = 0;
    int county = 0;
    for(int i = 0; i < nc; i++)
    {
        for(int j = 0; j < nh * nw; j++)
        {
            for(int k = 0; k < m; k++)
            {
                Vects[i]->access(j, k) = Imgs[k][i]->access(countx, county);
            }
            county++;
            if(county > nh - 1)
            {
                county = 0;
                countx++;
            }
        }
        countx = 0;
        county = 0;
    }

    return Vects;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Volume Vects2Imgs(Matrix* Vects)
{
    int nh = sqrt(Vects->Rows());
    int nw = nh;
    int m = Vects->Columns();

    Volume Imgs(m, nh, nw);
    int countx = 0;
    int county = 0;
    Matrix* temp = Vects->TRANSPOSE();
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < nh; j++)
        {
            for(int k = 0; k < nw; k++)
            {
                Imgs[i]->access(j, k) = temp->access(countx, county);
                county++;
            }
        }
        county = 0;
        countx++;
    }
    delete temp;

    return Imgs;
}


VectVolume convolve(VectVolume Aprev, VectVolume filters, int stride)
{
    Matrix* convTemp = nullptr;
	Matrix* FilterMat = nullptr;
	Matrix* convAccum = nullptr;
	Matrix* Matptr = nullptr;

    int m = Aprev.size();
    int nh = Aprev[0][0]->Rows();
    int nw = Aprev[0][0]->Columns();
    int numOfFilters = filters.size();
    int filter_nc = filters[0].size();

    Volume a_prev = Imgs2Vects(Aprev);

    VectVolume Result(m, numOfFilters);

    for(int i = 0; i < numOfFilters; i++)
    {
        for(int k = 0; k < filter_nc; k++)
        {
            FilterMat = FilterToMatrix(filters[i][k], nh, nw, stride);
            //convTemp = DOT(FilterMat,a_prev[k]);
            convTemp = FilterMat->dot_T(a_prev[k]);
            delete FilterMat;
            if(k == 0)
                convAccum = convTemp;
            else
            {
                Matptr = convAccum;
                convAccum = convAccum->add(convTemp);
                delete Matptr;
                delete convTemp;
            }
        }

        Volume Images = Vects2Imgs(convAccum);
        for(int k = 0; k < m; k++)
        {
            Result[k][i] = Images[k];
        }
    }

    a_prev.DELETE();
    return Result;
}


VectVolume convolve_Threaded(VectVolume Aprev, VectVolume filters, int stride)
{
    Matrix* convTemp = nullptr;
	Matrix* FilterMat = nullptr;
	Matrix* convAccum = nullptr;
	Matrix* Matptr = nullptr;

    int m = Aprev.size();
    int nh = Aprev[0][0]->Rows();
    int nw = Aprev[0][0]->Columns();
    int numOfFilters = filters.size();
    int filter_nc = filters[0].size();

    Volume a_prev = Imgs2Vects(Aprev);

    VectVolume Result(m, numOfFilters);

    for(int i = 0; i < numOfFilters; i++)
    {
        for(int k = 0; k < filter_nc; k++)
        {
            FilterMat = FilterToMatrix(filters[i][k], nh, nw, stride);
            convTemp = DOT(FilterMat,a_prev[k]);
            //convTemp = FilterMat->dot_T(a_prev[k]);
            delete FilterMat;
            if(k == 0)
                convAccum = convTemp;
            else
            {
                Matptr = convAccum;
                convAccum = convAccum->add(convTemp);
                delete Matptr;
                delete convTemp;
            }
        }

        Volume Images = Vects2Imgs(convAccum);
        for(int k = 0; k < m; k++)
        {
            Result[k][i] = Images[k];
        }
        delete convAccum;
    }

    a_prev.DELETE();
    return Result;
}


Matrix* convolve2(Volume& Aprev, Volume& Filter, int s)
{

    Matrix* convTemp = nullptr;
	Matrix* Matptr1 = nullptr;
	Matrix* Matptr2 = nullptr;
	Matrix* filter = nullptr;

    int nh=Aprev[0]->Rows();
    int nw=Aprev[0]->Columns();
    int filter_nc = Filter.size();


    Matrix* a_prev = to_1D(Aprev);
    for(int k = 0; k < filter_nc; k++)
    {
        filter  = FilterToMatrix(Filter[k], nh, nw, s);
        Matptr1 = a_prev->SubMat(0, k, -1, k);
        Matptr2 = filter->dot_T(Matptr1);
        delete filter;
        delete Matptr1;
        if(k == 0)
        {
            convTemp = to_Mat(Matptr2);
            delete Matptr2;
        }
        else
        {
            Matptr1 = Matptr2;
            Matptr2 = to_Mat(Matptr2);
            delete Matptr1;
            Matptr1 = convTemp;
            convTemp = convTemp->add(Matptr2);
            delete Matptr2;
            delete Matptr1;
        }
    }
    delete a_prev;

    return convTemp;
}

Matrix* to_Mat(Matrix* X)
{
	int dim = sqrt(X->Rows());
    Matrix* X_2D = new Matrix(dim, dim);
    for(int i=0; i < dim; i++)
        for (int j = 0; j < dim; j++)
        {
            X_2D->data[i][j] = X->data[i*dim + j][0];
        }
	return X_2D;
}
