#include "NN_Tools.h"
using namespace std;
////////////////////////////////////////////////////////////////////////////////////
string CharGen(string name, int i)
{
    int temp = i;
    int counter1;   //number of decimal digits in i

	if (temp == 0)
		counter1 = 1;
	else
	{
		for (counter1 = 0; temp != 0; counter1++)
			temp = temp / 10;
	}


    int counter2=name.size();   //number of chars in name

    string result;
    if (counter2 == 1)  { result = "W0";}
    if (counter2 == 2)  { result = "dW0";}
    if (counter2 == 3)  { result = "Sdw0";}
    if (counter2 == 4)  { result = "dACP0";}
	if (counter2 == 5)  { result = "dACP01"; }
	if (counter2 == 6)  { result = "dACP012"; }
	if (counter2 == 7)  { result = "dACP0123"; }
	if (counter2 == 8)  { result = "dACP01234"; }
	if (counter2 == 9)  { result = "dACP012345"; }
	if (counter2 == 10) { result = "dACP0123456"; }
	if (counter2 == 11) { result = "dACP01234567"; }
	if (counter2 == 12) { result = "dACP012345678"; }


    for (unsigned int j = 0; j<name.size(); j++) //copy the name into result
        result[j] = name[j];

    int j = counter1 + counter2 - 1;      //copy the number into result
    temp = i;
    do
    {
        result[j] = '0' + (temp % 10);
        temp = temp / 10;
        j--;
    }while (temp != 0);

    return result;
}
////////////////////////////////////////////////////////////////////////////////////
void AccuracyTest(Matrix* Y, Matrix* Y_hat, string devOrtest)
{
	float errSum = 0;
	Matrix* errors = new Matrix(10,1);
	for (int j = 0; j<Y_hat->Columns(); j++)
	{
		float maximum = Y_hat->access(0, j);
		int index = 0;
		for (int i = 1; i<Y_hat->Rows(); i++)
		{
			if (Y_hat->access(i, j)>maximum)
			{
				maximum = Y_hat->access(i, j);
				index = i;
			}
		}

		if (Y->access(index, j) != 1)
        {
            float maximum = Y->access(0, j);
            int ind = 0;
            for (int i = 1; i<Y->Rows(); i++)
            {
                if (Y->access(i, j)>maximum)
                {
                    maximum = Y->access(i, j);
                    ind = i;
                }
            }
            errors->access(ind,0)++;
        }
	}
	errSum=errors->sumall();
	float Accur = 1 - ((errSum) / Y->Columns());
	cout << "False Predictions = " << errSum << endl;
	if (devOrtest == "dev")
		cout <<"Dev Accuracy = " << Accur * 100 << "%" << endl;
	else if (devOrtest == "test")
		cout <<"Test Accuracy = "<< Accur * 100 << "%" << endl;
	cout<<"Errs : ";
	for(int i = 0; i < 10; i++)
        cout<<"["<< i <<"]="<<errors->access(i,0)<<" ";
    cout<<endl;
	delete errors;
}
/////////////////////////////////////////////////////////////////////////////////
void AccuracyTest(Matrix* Y, Matrix* Y_hat, Matrix* errors)
{
	for (int j = 0; j<Y_hat->Columns(); j++)
	{
		float maximum = Y_hat->access(0, j);
		int index = 0;
		for (int i = 1; i<Y_hat->Rows(); i++)
		{
			if (Y_hat->access(i, j)>maximum)
			{
				maximum = Y_hat->access(i, j);
				index = i;
			}
		}

		if (Y->access(index, j) != 1)
        {
            float maximum = Y->access(0, j);
            int ind = 0;
            for (int i = 1; i<Y->Rows(); i++)
            {
                if (Y->access(i, j)>maximum)
                {
                    maximum = Y->access(i, j);
                    ind = i;
                }
            }
            errors->access(ind,0)++;
        }
	}
}
///////////////////////////////////////////////////////////////////////
Matrix* DOT(Matrix* X, Matrix* Y)
{
	Matrix* result = new Matrix(X->Rows(),Y->Columns());
	int CORES = thread::hardware_concurrency();
	thread** Threads = new  thread* [CORES];
	Y = Y->TRANSPOSE();
	for(int i=0; i<CORES; i++)
	{
	    Threads[i] = new thread(DotPart,i+1,result,X,Y);
	}

	for (int i = 0; i<CORES; i++)
	{
		Threads[i]->join();
		delete Threads[i];
	}
	delete Threads;

	if(X->row % CORES != 0)
    {
        int numOfRows = X->row % CORES;
        int limit = X->row;
        int start = limit - numOfRows;
		for (int i = start; i < limit; i++)
			for (int j = 0; j < Y->row; j++)
				for (int k = 0; k < X->column; k++)
					result->data[i][j] += X->data[i][k] * Y->data[j][k];
    }
    delete Y;
	return result;
}
///////////////////////////////////////////////////////////////////////
void DotPart(int part, Matrix* result, Matrix* X, Matrix* Y)
{
    int numOfRows = X->row / thread::hardware_concurrency();
	int limit = part*numOfRows;
	int start = limit - numOfRows;
		for (int i = start; i < limit; i++)
			for (int j = 0; j < Y->row; j++)
				for (int k = 0; k < X->column; k++)
					result->data[i][j] += X->data[i][k] * Y->data[j][k];
}
///////////////////////////////////////////////////////////////////////
void unlearned_patterns(int& ncols, Matrix* Y, Matrix* Y_hat, int*indices, int minibatch_num, int minibatch_size, float StudiedWell )
{
    //when should we consider a patter is learnt well? by well we may consider the maximum estimation must be larger than 0.8 (StudiedWell)
    for (int j = 0; j<Y_hat->Columns(); j++)
	{
	    //get the maximum estimation and its index
		float maximum = Y_hat->access(0, j);
		int index = 0;
		for (int i = 1; i<Y_hat->Rows(); i++)
		{
			if (Y_hat->access(i, j) > maximum)
			{
				maximum = Y_hat->access(i, j);
				index = i;
			}
		}

		if ((Y->access(index, j) != 1) || (maximum < StudiedWell) )//
        {
            indices[ncols]=j;
			ncols++;
        }
	}
	if(ncols < 2)
        cout<<"WARINING"<<ncols<<endl;
}
/////////////////////////////////////////////////////////////////////////////////////////////
Matrix* reduced_X(Matrix* X, int* indices, int ncols)
{
    Matrix* XX = new Matrix(X->Rows(),ncols);
     for(int j=0; j<X->Rows(); j++)
       {
           for(int k=0; k<ncols; k++)
           {
                XX->access(j,k) = X->access(j,indices[k]);
           }
       }

    return XX;
}
/////////////////////////////////////////////////////////////////////////////////////////////
Matrix* reduced_Y(Matrix*Y,int* indices,int ncols)
{
    Matrix* YY = new Matrix(Y->Rows(),ncols);
     for(int j=0;j<Y->Rows();j++)
       {
           for(int k=0;k<ncols;k++)
           {
                YY->access(j,k)=Y->access(j,indices[k]);

           }
       }
   return YY;
}
//////////////////////////////////////////////////////////////////////////////////////////////
void getIndices(Matrix* Y, int pattern, int* indices, int& numOfPatterns)
{
	for (int j = 0; j<Y->Columns(); j++)
	{
		if (Y->data[pattern][j] == 1)
		{
			indices[numOfPatterns] = j;
			numOfPatterns++;
		}
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////
void getPattern(Volume& X_pattern, Matrix*& Y_pattern, Matrix* X, Matrix* Y, int pattern)
{
	if (pattern < 0 || pattern > 9)
	{
		cout << "false pattern" << endl;
		return;
	}

	Matrix* XX = nullptr;
	int indices[60000];
	int numOfPatterns = 0;
	getIndices(Y, pattern, indices, numOfPatterns);
	XX = reduced_X(X, indices, numOfPatterns);
	Y_pattern = reduced_Y(Y, indices, numOfPatterns);
	X_pattern = to_2D(XX);
	delete XX;
}
///////////////////////////////////////////////////////////////////////
void MIX(Matrix*& X, Matrix*& Y, Matrix* X_, Matrix* Y_)
{
    Matrix* XX=new Matrix(X->row,X->column+X_->column);
    Matrix* YY=new Matrix(Y->row,Y->column+Y_->column);
    int i=0;
    int j=0;
    int k=0;
    int m=0;

    for(i=0; i<X->row; i++)
        for(j=0; j<X->column; j++)
            XX->data[i][j]=X->data[i][j];


    for(i=0; i<XX->row; i++)
        for(k=j,m=0; k<XX->column; k++,m++)
            XX->data[i][k]=X_->data[i][m];




    for(i=0; i<Y->row; i++)
        for(j=0; j<Y->column; j++)
            YY->data[i][j]=Y->data[i][j];


    for(i=0; i<YY->row; i++)
        for(k=j,m=0; k<YY->column; k++,m++)
            YY->data[i][k]=Y_->data[i][m];


    delete X;
    delete Y;
    delete X_;
    delete Y_;
    X=XX;
    Y=YY;
}

