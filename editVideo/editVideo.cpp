#include <iostream>
#include <fstream>
#include <string>
using namespace std;

#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#define ratio 200
int main(int argc , char** argv)
{

cv::VideoCapture capture(argv[1]);

if(!capture.isOpened())
	return 1;

double rate = capture.get(CV_CAP_PROP_FPS);
cout << "Rate of the video is " << rate;

bool stop(false);

cv::Mat frame;
cv::namedWindow("Extracted Frame");

int delay = 1000/rate;

ofstream FrameRecord("/home/jfoucs/MYGraduationProject/Data/associate.txt");


while(!stop)
{
	static int FrameNum;
	FrameNum++;
	if(!capture.read(frame))
		break;
//	cv::imshow("Extracted Frame",frame);
	if( !(FrameNum % ratio) )
	{
		cv::imwrite("/home/jfoucs/MYGraduationProject/FrameExtract/" + std::to_string( (FrameNum/ratio) ) + ".png" , frame);
		if(FrameRecord.is_open())
			{
				FrameRecord << std::to_string((FrameNum/ratio)) << " " << "/home/jfoucs/MYGraduationProject/FrameExtract/" << std::to_string((FrameNum/ratio)) << ".png" << "\n";
			}
	}
	
	
//	if( cv::waitKey(delay)>=0 )
//		stop = true;

}

FrameRecord.close();
capture.release();
return 0;


}
