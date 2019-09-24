#include "Activations.h"
float ActivPar=0.8;

Matrix* activ(Matrix*z, ActivationType activtype)
{
	Matrix* A = nullptr;
	switch (activtype)
	{
	case RELU:
		A = relu(z); break;
	case LEAKYRELU:
		A = leakyRelu(z); break;
	case TANH:
		A = mytanh(z); break;
	case SIGMOID:
		A = sigmoid(z); break;
	case SOFTMAX:
		A = softmax(z); break;
	case SATLINEAR:
		A = satLinear(z); break;
	case LINEAR:
		A = Linear(z); break;
    case SATLINEAR2:
        A = satLinear2(z, 0); break;
    case SATLINEAR3:
        A = satLinear3(z, 0); break;
    case NONE:
        break;
	}
	return A;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* dactiv(Matrix* z, ActivationType activtype)
{
	Matrix* df = nullptr;
	switch (activtype)
	{
	case RELU:
		df = drelu(z); break;
	case LEAKYRELU:
		df = dleakyRelu(z); break;
	case TANH:
		df = dtanh(z); break;
	case SIGMOID:
		df = dsigmoid(z); break;
	case SOFTMAX:
		df = dsoftmax(z); break;
	case SATLINEAR:
		df = dsatLinear(z); break;
	case LINEAR:
		df = dLinear(z); break;
    case SATLINEAR2:
        df = dsatLinear2(z, 0); break;
    case SATLINEAR3:
        df = dsatLinear3(z, 0); break;
    case NONE:
        break;
	}
	return df;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* softmax(Matrix* z)
{
    float MAX=z->MaxElement();
    Matrix* z_stable = z->sub(MAX);  /////
    float epsi=1e-10;
    int r=z->Rows();
    int c=z->Columns();
    Matrix* A = new Matrix(r,c); /////
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            A->access(i,j) = exp(z_stable->access(i,j));
        }
    }
    Matrix* s =A->SUM("row");                   //summation of exponents
    for(int ii=0;ii<r;ii++)
    {
        for(int jj=0;jj<c;jj++)
        {
            A->access(ii,jj) = A->access(ii,jj)/(s->access(0,jj)+epsi);
        }
    }
    delete z_stable;
    delete s;
    return A;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* sigmoid(Matrix* z)
{
    int r=z->Rows();
    int c=z->Columns();
    Matrix* A=new Matrix(r,c);
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            A->access(i,j) = 1/(1+exp(-1*z->access(i,j)));
        }
    }
    return A;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* mytanh(Matrix* z)
{
    int r=z->Rows();
    int c=z->Columns();
    Matrix* A=new Matrix(r,c);
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            A->access(i,j)=tanh(z->access(i,j));
        }
    }
    return A;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* relu(Matrix* z)
{
    int r=z->Rows();
    int c=z->Columns();
    Matrix* A=new Matrix(r,c);
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            if(z->access(i,j)>=0)
            A->access(i,j)=z->access(i,j);
            else
            A->access(i,j)=0;
        }
    }
    return A;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* leakyRelu(Matrix* z)
{
    int r=z->Rows();
    int c=z->Columns();
    Matrix* A=new Matrix(r,c);
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            if(z->access(i,j)>=0)
                A->access(i,j)=z->access(i,j);
            else
                A->access(i,j)=0.01*z->access(i,j);
        }
    }
    return A;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* Linear(Matrix* z)
{
    return z;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* satLinear(Matrix* z)
{
    int r=z->Rows();
    int c=z->Columns();
    Matrix* A=new Matrix(r,c);
    for(int i=0; i<r; i++)
    {
        for(int j=0; j<c; j++)
        {
            if(z->access(i,j)<=-1)
                 A->access(i,j)=-1;
            else if(z->access(i,j)>-1 && z->access(i,j)<1)
                 A->access(i,j)=1*z->access(i,j);
            else
                 A->access(i,j)=1;
        }
    }
    return A;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* satLinear2(Matrix* z, float maxErr)
{
    float slope;

    if(maxErr>=ActivPar)
        slope=1;
    else
        slope=2;

    int r=z->Rows();
    int c=z->Columns();
    Matrix* A=new Matrix(r,c);

    for(int i=0; i<r; i++)
    {
        for(int j=0; j<c; j++)
        {
            if(z->access(i,j)<=-1/slope)
                 A->access(i,j)=-1;
            else if(z->access(i,j)>-1/slope && z->access(i,j)<1/slope)
                 A->access(i,j)=slope*z->access(i,j);
            else
                 A->access(i,j)=1;
        }
    }

    return A;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* satLinear3(Matrix* z,float maxErr)
{
    float slope;

    if(maxErr>=ActivPar)
        slope=1;
    else
        slope=2;

    int r=z->Rows();
    int c=z->Columns();
    Matrix* A=new Matrix(r,c);

    for(int i=0; i<r; i++)
    {
        for(int j=0; j<c; j++)
        {
            if(z->access(i,j)<=-1/slope)
                 A->access(i,j)=-1;
            else if(z->access(i,j)>-1/slope && z->access(i,j)<1/slope)
                 A->access(i,j)=slope*z->access(i,j);
            else
                 A->access(i,j)=1;
        }
    }

    return A;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* dsoftmax(Matrix* z)                        //da=a(1-a)
{
    Matrix* a=softmax(z);
    Matrix* t1=a->mul(-1);
    Matrix* t2=t1->add(1);
    Matrix* da=t2->mul(a);
    delete a;
    delete t1;
    delete t2;
    return da;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* dsigmoid(Matrix* z)                      //da=a(1-a)
{
    Matrix* a=sigmoid(z);
    Matrix* t1=a->mul(-1);
    Matrix* t2=t1->add(1);
    Matrix* da=t2->mul(a);
    delete a;
    delete t1;
    delete t2;
    return da;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* dtanh(Matrix* z)                                 //da=1-a^2
{
    Matrix* a=mytanh(z);
    Matrix* t1=a->mul(a);
    Matrix* t2=t1->mul(-1);
    Matrix* da=t2->add(1);
    delete a;
    delete t1;
    delete t2;
    return da;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* drelu(Matrix* z)
{
    int r=z->Rows();
    int c=z->Columns();
    Matrix* da=new Matrix(r,c);
    for(int i=0;i<r;i++)
    {
        for (int j=0;j<c;j++)
        {
            if(z->access(i,j)<0)
            da->access(i,j)=0;
            else
            da->access(i,j)=1;
        }
    }
    return da;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* dleakyRelu(Matrix* z)
{
    int r=z->Rows();
    int c=z->Columns();
    Matrix* da=new Matrix(r,c);
    for(int i=0;i<r;i++)
    {
        for (int j=0;j<c;j++)
        {
            if(z->access(i,j)<0)
                da->access(i,j)=0.01;
            else
                da->access(i,j)=1;
        }
    }
    return da;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* dLinear(Matrix* z)
{
    int r=z->Rows();
    int c=z->Columns();
    Matrix* da=new Matrix(r,c,1);
    return da;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* dsatLinear(Matrix* z)
{
    int r=z->Rows();
    int c=z->Columns();
    Matrix* da=new Matrix(r,c);
    for(int i=0;i<r;i++)
    {
        for (int j=0;j<c;j++)
        {
            if(z->access(i,j)<=-1)
                 da->access(i,j)=0;
            else if(z->access(i,j)>-1 && z->access(i,j)<1)
                 da->access(i,j)=1;
            else
                 da->access(i,j)=0;
        }
    }
    return da;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* dsatLinear2(Matrix* z,float maxErr)
{
    float slope;

    if(maxErr>=ActivPar)
        slope=1;
    else
        slope=2;

    int r=z->Rows();
    int c=z->Columns();
    Matrix* da=new Matrix(r,c);
    for(int i=0;i<r;i++)
    {
        for (int j=0;j<c;j++)
        {
            if(z->access(i,j)<=-1/slope)
                 da->access(i,j)=0;

            else if(z->access(i,j)>-1/slope && z->access(i,j)<1/slope)
                 da->access(i,j)=slope;

            else
                 da->access(i,j)=0;
        }
    }
    return da;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* dsatLinear3(Matrix* z, float maxErr)
{
    float slope;

    if(maxErr>=ActivPar)
        slope=1;
    else
        slope=2;

    int r=z->Rows();
    int c=z->Columns();
    Matrix* da=new Matrix(r,c);
    for(int i=0;i<r;i++)
    {
        for (int j=0;j<c;j++)
        {
            if(z->access(i,j)<=-1/slope)
                 da->access(i,j)=0;

            else if(z->access(i,j)>-1/slope && z->access(i,j)<1/slope)
                 da->access(i,j)=slope;

            else
                 da->access(i,j)=0;
        }
    }
    return da;
}
