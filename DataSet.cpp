#include "DataSet.h"

///////////////////////////////////////////////////////////////////////////////
////////////////////////////GET DATASET FROM HARD DISK/////////////////////////
///////////////////////////////////////////////////////////////////////////////
int LittleEndian(uint32_t ii)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=ii&255;
    ch2=(ii>>8)&255;
    ch3=(ii>>16)&255;
    ch4=(ii>>24)&255;
   return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+int(ch4);
}
///////////////////////////////////////////////////////////////////////////////
void get_dataset(Matrix* X,Matrix* Y,const char*Xdir,const char*Ydir,uint32_t EXAMPLES)
{
    //X (28*28,ImagesNUM)
    //Y (10,ImagesNUM)
    cout<<endl<<">> MNIST Test Set:"<<endl;
    ifstream pixels(Xdir,ios::binary);
    uint32_t magicNum1;
    uint32_t ImagesNum1;
    uint32_t RowsNum;
    uint32_t ColumnsNum;
    pixels.read((char*)&magicNum1,sizeof(magicNum1));
    magicNum1=LittleEndian(magicNum1);
    pixels.read((char*)&ImagesNum1,sizeof(ImagesNum1));
    ImagesNum1=LittleEndian(ImagesNum1);
    cout<<"Number of images of Data = "<<ImagesNum1<<endl;
    pixels.read((char*)&RowsNum,sizeof(RowsNum));
    RowsNum=LittleEndian(RowsNum);
    cout<<"Number of Rows = "<<RowsNum<<endl;
    pixels.read((char*)&ColumnsNum,sizeof(ColumnsNum));
    ColumnsNum=LittleEndian(ColumnsNum);
    cout<<"Number of Columns = "<<ColumnsNum<<endl;

    ifstream labels(Ydir,ios::binary);
    uint32_t magicNum2;
    uint32_t ImagesNum2;
    labels.read((char*)&magicNum2,sizeof(magicNum2));
    magicNum2=LittleEndian(magicNum2);
    labels.read((char*)&ImagesNum2,sizeof(ImagesNum2));
    ImagesNum2=LittleEndian(ImagesNum2);


    for (uint32_t j=0;j<EXAMPLES;j++)
    {
        for(uint32_t i=0;i<RowsNum*ColumnsNum;i++)
        {
           unsigned char temp1;
           pixels.read((char*)&temp1,1);
           X->access(i,j)=temp1/255.0;
        }

        unsigned char temp2;
        labels.read((char*)&temp2,1);
        Y->access(int(temp2),j)=1;
    }

}
///////////////////////////////////////////////////////////////////////////////
void Shuffle(Matrix* X, Matrix* Y)
{
	void SWAP(Matrix* MAT, int i, int k);

	for (int i = 0; i < X->Columns(); i++)
	{
		int s = rand() % X->Columns();
		SWAP(X, i, s);
		SWAP(Y, i, s);
	}

}
///////////////////////////////////////////////////////////////////////////////
int get_dataset_2D(Volume& X, Matrix* Y,const char*Xdir,const  char*Ydir,uint32_t EXAMPLES)
{
	//X Volume(60000)
	//Y (10,ImagesNUM)
	cout<<">> MNIST Training Set:"<<endl;
	ifstream pixels(Xdir,ios::binary);
    uint32_t magicNum1;
    uint32_t ImagesNum1;
    uint32_t RowsNum;
    uint32_t ColumnsNum;
    pixels.read((char*)&magicNum1,sizeof(magicNum1));
    magicNum1=LittleEndian(magicNum1);
    pixels.read((char*)&ImagesNum1,sizeof(ImagesNum1));
    ImagesNum1=LittleEndian(ImagesNum1);
    cout<<"Number of images of Data = "<<ImagesNum1<<endl;
    pixels.read((char*)&RowsNum,sizeof(RowsNum));
    RowsNum=LittleEndian(RowsNum);
    cout<<"Number of Rows = "<<RowsNum<<endl;
    pixels.read((char*)&ColumnsNum,sizeof(ColumnsNum));
    ColumnsNum=LittleEndian(ColumnsNum);
    cout<<"Number of Columns = "<<ColumnsNum<<endl;

	ifstream labels(Ydir, ios::binary);
	uint32_t magicNum2;
	uint32_t ImagesNum2;
	labels.read((char*)&magicNum2, sizeof(magicNum2));
	magicNum2 = LittleEndian(magicNum2);
	labels.read((char*)&ImagesNum2, sizeof(ImagesNum2));
	ImagesNum2 = LittleEndian(ImagesNum2);


	for (uint32_t k = 0; k<EXAMPLES; k++)
	{
		X[k] = new Matrix(RowsNum, ColumnsNum, 0);
		for (uint32_t i = 0; i < RowsNum; i++)
			for (uint32_t j = 0; j < ColumnsNum; j++)
			{
				unsigned char temp1;
				pixels.read((char*)&temp1, 1);
				X[k]->access(i,j)= temp1;
			}

		unsigned char temp2;
		labels.read((char*)&temp2, 1);
		Y->access(int(temp2), k) = 1;
	}
	return 0;
}
///////////////////////////////////////////////////////////////////////////////
void Shuffle(Volume& X, Matrix* Y)
{
	void SWAP(Matrix* MAT, int i, int k);
	void SWAP(Volume& Vol, int i, int k);

	for (int i = 0; i<Y->Columns(); i++)
	{
		int s = rand() % Y->Columns();
		SWAP(X, i, s);
		SWAP(Y, i, s);
	}

}
///////////////////////////////////////////////////////////////////////////////
void SWAP(Matrix* MAT, int i, int k)
{
	Matrix* temp = new Matrix(MAT->Rows(), 1);
	for (int j = 0; j<MAT->Rows(); j++)
	{
		temp->data[j][0] = MAT->data[j][i];
		MAT->data[j][i] = MAT->data[j][k];
		MAT->data[j][k] = temp->data[j][0];
	}
	delete temp;
}
///////////////////////////////////////////////////////////////////////////////
void SWAP(Volume& Vol, int i, int k)
{
	Matrix* temp;
	temp = Vol[i];
	Vol[i] = Vol[k];
	Vol[k] = temp;
}
///////////////////////////////////////////////////////////////////////////////
void DevSet(Matrix* X, Matrix* Y, Matrix* X_dev, Matrix* Y_dev, int DEV)
{
    for(int j=0; j<DEV; j++)
        for(int i=0; i<X->Rows(); i++)
            X_dev->access(i,j)=X->access(i,j);


    for(int j=0; j<DEV; j++)
        for(int i=0; i<Y->Rows(); i++)
        {
            Y_dev->access(i,j)=Y->access(i,j);
        }

}
///////////////////////////////////////////////////////////////////////////////
void normalize(Matrix& X, Matrix& X_test, Matrix& Y, Matrix& Y_test)
{
    Matrix mean(784,1);
    Matrix variance(784,1);
    float eps=1e-8;


    mean=(X.sum("column")+X_test.sum("column"))/(X.Columns()+X_test.Columns());
    X=X-mean;
    X_test=X_test-mean;
    variance=(X.square().sum("column")+X_test.square().sum("column"))/(X.Columns()+X_test.Columns());
    variance=variance+eps;
    X=X/(variance.Sqrt());
    X_test=X_test/(variance.Sqrt());

    /*for(int i=0; i<Y.Rows(); i++)
        for(int j=0; j<Y.Columns(); j++)
        {
            if(Y.access(i,j)==0)
                Y.access(i,j)=-1;
        }


   for(int i=0; i<Y_test.Rows(); i++)
        for(int j=0; j<Y_test.Columns(); j++)
        {
            if(Y_test.access(i,j)==0)
                Y_test.access(i,j)=-1;
        }*/
}
///////////////////////////////////////////////////////////////////////////////
//////////////////////////ELASTIC DISTORTION///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/*returns a gaussian filter (x,y) with standard deviation sigma*/
Matrix* gausianFilter(int x, int y, float sigma)
{
	Matrix* GaussianFilter = new Matrix(x, y);
	float sum = 0.0;
	int xx = (x - 1) / 2;
	int yy = (y - 1) / 2;

	for (int i = 0; i<x; i++)
	{
		for (int j = 0; j<y; j++)
		{
			int ii = (i - xx);
			int jj = (j - yy);
			float r = sqrt(ii*ii + jj * jj);
			GaussianFilter->access(i, j) = exp(-(r*r) / (2 * sigma*sigma)) / (2 * 3.14159265358979323846*sigma*sigma);
			sum += GaussianFilter->access(i, j);
		}
	}

	for (int i = 0; i<x; i++)
	{
		for (int j = 0; j<y; j++)
		{
			GaussianFilter->access(i, j) = GaussianFilter->access(i, j) / sum;
		}
	}

	return GaussianFilter;
}
/////////////////////////////////////////////////////////////////////////////////////////////////

/*perform a gaussian blur to a 2D img*/
Matrix* gaussianBlur(Matrix* img, int filterSize, float sigma)
{
	int p = (filterSize - 1) / 2;

	Matrix* paddedImg = pad(img, p, 0);

	Matrix* filter = gausianFilter(filterSize, filterSize, sigma);

	Volume Img(1);
	Img[0] = paddedImg;

	Volume Filter(1);
	Filter[0] = filter;


	Matrix* newImg= convolve2(Img, Filter, 1);


	delete filter;
	delete paddedImg;
	return newImg;
}
/////////////////////////////////////////////////////////////////////////////////////////////////

/*L1 norm to matrix x*/
float norm_L1(Matrix* x)
{
	float sum = 0;
	for (int i = 0; i<x->Rows(); i++)
		for (int j = 0; j < x->Columns(); j++)
		{
			if (x->access(i, j) < 0)
				sum = sum - x->access(i, j);
			else
				sum = sum + x->access(i, j);
		}
	return sum;
}
/////////////////////////////////////////////////////////////////////////////////////////////////


/*performs elastric distortion on an img*/
Matrix* elasticDistortion(Matrix* img, int filterSize, float sigma, float alpha)
{
    //temporal ptrs
    Matrix* Matptr1 = nullptr;
    Matrix* Matptr2 = nullptr;

	//displacement matrices dx & dy with uniform random values in range -1,1
	Matrix* dx = new Matrix(img->Rows(), img->Columns(), Random);
	Matrix* dy = new Matrix(img->Rows(), img->Columns(), Random);
	Matrix* dx_old = nullptr;
	Matrix* dy_old = nullptr;


	//dx = dx * (2/RAND_MAX) - 1
	Matptr1 = dx;
	Matptr2 = dx->mul(2.0/float(RAND_MAX));
	dx = Matptr2->sub(1);
	delete Matptr1;
	delete Matptr2;

	//dy = dy * (2/RAND_MAX) - 1
	Matptr1 = dy;
	Matptr2 = dy->mul(2.0/float(RAND_MAX));
	dy = Matptr2->sub(1);
	delete Matptr1;
	delete Matptr2;



	//apply gaussian filter to the displacements
	dx_old = dx;
	dy_old = dy;
	dx = gaussianBlur(dx, filterSize, sigma);
	dy = gaussianBlur(dy, filterSize, sigma);


	//normalizing dx & dy (dx = dx / norm(dx)) (dy = dy / norm(dy))
	Matptr1 = dx;
	dx = dx->div(norm_L1(dx));
	delete Matptr1;
	Matptr1 = dy;
	dy = dy->div(norm_L1(dy));
	delete Matptr1;



	//alpha controls the intensity of deformation (dx = dx * alpha) (dy = dy * alpha)
	Matptr1 = dx;
	dx = dx->mul(alpha);
	delete Matptr1;
	Matptr1 = dy;
	dy = dy->mul(alpha);
	delete Matptr1;


	//apply displacements, we assume the top left corner is position (0,0) and bottom right corner is position (img->Rows()-1 , img->Columns()-1)
	Matrix* distImg = new Matrix(img->Rows(), img->Columns());

	for (int i = 0; i<img->Rows(); i++)
		for (int j = 0; j < img->Columns(); j++)
		{
			//the position of the new pixel value
			float x = i + dx->access(i, j); //x=0+1.75=1.75
			float y = j + dy->access(i, j); //y=0+0.5=0.5

											//if the new position is outside the image put 0 in it
			if (x < 0 || y < 0)
			{
				distImg->access(i, j) = 0;
			}
			else
			{
				//applying bilinear interpolation to the unit square with (xmin,ymin) (xmax,ymax), (xdis,ydis) is a point in the unit square
				int xmin = int(x);
				int xmax = xmin + 1;
				int ymin = int(y);
				int ymax = ymin + 1;
				float xdis = x - int(x);
				float ydis = y - int(y);

				if (xmin >= img->Columns() || xmax >= img->Columns() || ymin >= img->Rows() || ymax >= img->Rows())
				{
					distImg->access(i, j) = 0;
				}
				else
				{
					//getting the pixels of current square
					float topLeft = img->access(xmin, ymin);
					float bottomLeft = img->access(xmin, ymax);
					float topRight = img->access(xmax, ymin);
					float bottomRight = img->access(xmax, ymax);

					//horizontal interpolation
					float horizTop = topLeft + xdis * (topRight - topLeft);
					if (horizTop < 0)
						horizTop = 0;
					float horizBottom = bottomLeft + xdis * (bottomRight - bottomLeft);
					if (horizBottom < 0)
						horizBottom = 0;

					//vertical interpolation
					float newPixel = horizTop + ydis * (horizBottom - horizTop);
					if (newPixel >= 0)
						distImg->access(i, j) = newPixel;
					else
						distImg->access(i, j) = 0;
				}
			}
		}

	delete dx;
	delete dy;
	delete dx_old;
	delete dy_old;
	return distImg;
}
/////////////////////////////////////////////////////////////////////////////////////////////////

/*enlarge the 2D dataset X with a factor enlargeFact*/
void enlarge2D(IntMatrix * X_1D, Volume& X, int SIZE, int enlargeFact, int threadnum)
{
	int CORES = std::thread::hardware_concurrency();
	int filterSize,sigma,alpha;
	Matrix* newImg = nullptr;

	for (int k = 0; k < enlargeFact; k++)
	{
		if (k == 0) { filterSize = 17; sigma = 10; alpha = 600; }
		if (k == 1) { filterSize = 19; sigma = 10; alpha = 600; }
		if (k == 2) { filterSize = 21; sigma = 10; alpha = 600; }
		if (k == 3) { filterSize = 23; sigma = 10; alpha = 600; }
		if (k == 4) { filterSize = 25; sigma = 10; alpha = 600; }
		if (k == 5) { filterSize = 27; sigma = 10; alpha = 600; }

		clock_t START = clock();
        if (threadnum == 0)
            cout<<"Creating X"<<k+1<<" ";
		for (int i = 0; i < SIZE / CORES; i++)
		{
			int num = threadnum*(SIZE / CORES) + i;
			newImg = elasticDistortion(X[num], filterSize, sigma, alpha);
			int col = SIZE + threadnum*(SIZE / CORES) + k*SIZE + i;
			to_1D(X_1D, newImg, col);
			delete newImg;
			if (i % 5 == 0 && threadnum == 0)
			{
				cout <<".";
			}
		}
		if (threadnum == 0)
            cout<<endl<<"X"<<k+1<<" Is Created In "<< (clock() - START) / CLOCKS_PER_SEC<<" Secs"<<endl;
	}
}



/////////////////////////////////////////////////////////////////////////////////////////////////

/*enlarge the 1D dataset X with a factor enlargeFact*/
IntMatrix* enlarge1D(Volume& X, Matrix*& Y, int enlargeFact)
{
    cout<<endl<<">> Enlarging The Training Set:"<<endl;
    //Copy the original images to the 1D matrix X_1D
	IntMatrix* X_1D = new IntMatrix(X[0]->Rows()*X[0]->Columns(), X.size()*(enlargeFact + 1));
	for (int i = 0; i<X.size(); i++)
	{

		to_1D(X_1D, X[i], i);
	}

	//Creating the large matrix Y_new to hold the labels for the new and old dataset
	Matrix* Y_new = new Matrix(Y->Rows(), Y->Columns() + Y->Columns()*enlargeFact);
	for (int k = 0; k < enlargeFact + 1; k++)
	{
		for (int i = 0; i < Y->Rows(); i++)
		{
			for (int j = 0; j < Y->Columns(); j++)
			{
				Y_new->access(i, k * Y->Columns() + j) = Y->access(i, j);
			}
		}
	}
	Matrix* temp = Y;
	Y = Y_new;
	delete temp;


	int SIZE = X.size();
	int coresnum = std::thread::hardware_concurrency();
	std::thread ** t = new  std::thread *[coresnum];

	for (int i = 0; i<coresnum; i++)
	{

		t[i] = new std::thread(enlarge2D, X_1D, std::ref(X),SIZE, enlargeFact, i);
	}

	for (int i = 0; i<coresnum; i++)
	{
		t[i]->join();
		delete t[i];
	}
	delete t;

	return X_1D;
}


/////////////////////////////////////////////////////////////////////////////////////////////////
void to_1D(IntMatrix* X_new, Matrix* X,int colNum)
{
    for(int i=0;i<X->Rows();i++)
    {
        for(int j=0;j<X->Columns();j++)
        {
            X_new->data[i*X->Rows()+j][colNum] = X->data[i][j];
        }
    }
}
