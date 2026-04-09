/* main.cpp */

//
// Performs a contrast stretch over a Windows bitmap (.bmp) file, making lighter pixels
// lighter and darker pixels darker.
//
// Usage: cs infile.bmp outfile.bmp steps
//
// << Anthony Milas >>
//
// Initial author:
//   Prof. Joe Hummel
//   Northwestern University
//

#include "app.h"


//
// Function prototypes:
//
uchar** DistributeImage(int myRank, int numProcs, uchar** image, int& rows, int& cols, int& rowsPerProc, int& leftOverRows);
uchar** CollectImage(int myRank, int numProcs, uchar** image, int rows, int cols, int rowsPerProc, int leftOverRows);


//
// main:
//
int main(int argc, char* argv[])
{
	char *infile;
	char *outfile;
	int   steps, myRank, numProcs;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);  // number of processes involved in run:
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);  	 // my proc id: 0 <= myRank < numProcs:

	//
	// process command-line args to program:
	//
	if (argc != 4)
	{
		cout << endl;
		cout << "Usage: mpiexec ... cs infile.bmp outfile.bmp steps" << endl;
		cout << endl;
		MPI_Abort(MPI_COMM_WORLD, 0 /*return code*/);
	}

	infile = argv[1];
	outfile = argv[2];
	steps = atoi(argv[3]);

	char host[128];

	gethostname(host, sizeof(host)/sizeof(host[0]));  // machine we are running on:

	cout << "process " << myRank 
	     << " starting on node '" << host << "'..." 
		 << endl;
	cout.flush();

	if (myRank == 0)
	{
		cout << endl;
		cout << "** Starting Contrast Stretch **" << endl;
		cout << "   Input file:  " << infile << endl;
		cout << "   Output file: " << outfile << endl;
		cout << "   Steps:       " << steps << endl;
		// cout << endl;
	}

	//
	// now let's input bitmap file:
	//
	BITMAPFILEHEADER bitmapFileHeader;
	BITMAPINFOHEADER bitmapInfoHeader;
	uchar** image = nullptr;
	int rows = 0, cols = 0, rowsPerProc = 0, leftOverRows = 0;

	if (myRank == 0)
	{
		//cout << "** Reading bitmap..." << endl;
		image = ReadBitmapFile(infile, bitmapFileHeader, bitmapInfoHeader, rows, cols);
		if (image == NULL)
		{
			cout << "** Failed to open image file, halting..." << endl;
			return 0;
		}

		cout << "   Image size:  "
		     << rows << " rows, "
			 << cols << " columns" << endl;
	    cout << endl;
	}

	//
	// okay, perform contrast stretching:
	//
	if (myRank == 0)
	{
		cout << "** Processing..." << endl;
	}

    auto start = chrono::high_resolution_clock::now();

	//
	// MASTER distributes matrix, WORKERS receives their chunk of matrix:
	//
	image = DistributeImage(myRank, numProcs, image, rows, cols, rowsPerProc, leftOverRows);

	//
	// Okay, everyone performs constrast-stretching on their chunk:
	//
	image = ContrastStretch(image, rows, cols, steps);

	//
	// Collect the results: WORKERS send, MASTER receives and puts image back together:
	//
	image = CollectImage(myRank, numProcs, image, rows, cols, rowsPerProc, leftOverRows);

    auto stop = chrono::high_resolution_clock::now();
    auto diff = stop - start;
    auto duration = chrono::duration_cast<chrono::milliseconds>(diff);

	//
	// Done, save image and output exec time:
	//
	if (myRank == 0)
	{
		cout << endl;
		cout << "** Done!  Time: " << duration.count() / 1000.0 << " secs" << endl;

		cout << "** Writing bitmap..." << endl;
		WriteBitmapFile(outfile, bitmapFileHeader, bitmapInfoHeader, image);

		cout << "** Execution complete." << endl;
		cout << endl;
	}

	//
	// done:
	//
	MPI_Finalize();

	return 0;
}


//
// DistributeImage: given the original image, rows and cols (from the master process),
// the master distributes this data to the worker processes.
//
// Upon return, the master has the entire image, but now rows x cols reflects the
// CHUNK the master should process.  For the workers, the image matrix contains
// their CHUNK of the matrix to process (this matrix includes room for ghost rows),
// along with the size of the matrix chunk (in rows and cols).
//
// NOTE: any extra rows (due to uneven split across processes) are kept by the master
// process; these extra rows are viewed as the start of the image.
//
uchar** DistributeImage(int myRank, int numProcs,
	                    uchar** image, int& rows, int& cols, 
						int& rowsPerProc, int& leftOverRows)
{
	int  params[2];

	// Master: distribute size of each worker's CHUNK (rows x cols), then the image data itself:
	if (myRank == 0)  // Master:
	{
		rowsPerProc = rows / numProcs;
		leftOverRows = rows % numProcs;
		rows = rowsPerProc + leftOverRows;  // okay, master is now only responsible for their own (the first) chunk:
		params[0] = rowsPerProc;
		params[1] = cols;
	}

	//Broadcasting image parameters
	MPI_Bcast(params, sizeof(params) / sizeof(params[0]), MPI_INT, 0, MPI_COMM_WORLD);

	if (myRank != 0) // Worker:
	{
		rows = params[0];
		cols = params[1];
		image = New2dMatrix<uchar>(rows + 2, cols * 3);  // workers need to create image matrix for CHUNK they will own, including ghost rows
	}

	//Scatter the image to the workers. Use MPI_IN_PLACE to Master doesn't receive a copy of original chunk
	uchar* sendbuf = (myRank == 0) ? image[leftOverRows] : NULL;
	uchar* recvbuf = (myRank == 0) ? (uchar*)MPI_IN_PLACE : image[1];
	MPI_Scatter(sendbuf, rowsPerProc * cols * 3, MPI_UNSIGNED_CHAR,
				recvbuf, rows * cols * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

	return image; // Done! Everyone returns back a matrix to process... (rows and cols should already be set for both master and workers)
}


//
// CollectImage: each worker sends in their image chunk (of size rows x cols), and 
// sends this chunk to the master.  The master collects these chunks and writes them
// back to the original image matrix.  Note that for all processes, rows and cols 
// denotes the size of their matrix CHUNK, not the original matrix size.
//
uchar** CollectImage(int myRank, int numProcs,
	                 uchar** image, int rows, int cols, 
	                 int rowsPerProc, int leftOverRows)
{
	// WORKERS send, MASTER receives:
	//Gather the image from the workers. Use MPI_IN_PLACE to Master doesn't receive a copy of original chunk
	uchar* sendbuf = (myRank == 0) ? (uchar*)MPI_IN_PLACE : image[1];
	uchar* recvbuf = (myRank == 0) ?  image[leftOverRows] : NULL;
	MPI_Gather(sendbuf, rows * cols * 3, MPI_UNSIGNED_CHAR,
			   recvbuf, rowsPerProc * cols * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

	if (myRank > 0)  // workers:
	{
		Delete2dMatrix<uchar>(image); // workers are done with CHUNK, so free associated memory:
		image = NULL;
	}

	return image; // Done, return final image
}
