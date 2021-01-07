#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <queue>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <functional>
#include <math.h>
#include <cmath>

using namespace std;
using namespace cv;

string get_input(ifstream& filename)
{
    stringstream stream;
    stream << filename.rdbuf();
    return stream.str();
}

class node
{
public:
    node* left;
    node* right;
    int character;
    int frequency;
    node(int character, int frequency) : character(character), frequency(frequency), left(nullptr), right(nullptr) {}
    node(int frequency, node* left, node* right) : character(0), frequency(frequency), left(left), right(right) {}
};

//node combine_leafs(pair<char, int> n1, pair<char, int> n2){}

//The third parameter for priority_queue is a compare class
class comparesecond
{
public:
    bool operator() (pair<int, int> n1, pair<int, int> n2)
    {
        return n1.second > n2.second;
    }
    bool operator() (node* n1, node* n2)
    {
        return n1->frequency > n2->frequency;
    }
};

//Given a map and a finished huffman tree, find the binary of each leaf node
//Make sure string here isn't a reference. Doing so, it will save the string even when recalled in recursion, it will just keep appending top existing binary
void encode(unordered_map<int, string>& unorderedmap, node* root, string binary)
{
    if (root == nullptr)
    {
        return;
    }
    if (root->left == nullptr && root->right == nullptr)
    {
        unorderedmap[root->character] = binary;
    }

    //DO NOT USE APPEND
    //We are adding extra things if we do
    encode(unorderedmap, root->left, binary + "0");
    encode(unorderedmap, root->right, binary + "1");
}

/*
void decode(unordered_map<string, char>& unorderedmap, node* root, const string& encoded)
{
    if (root == nullptr)
    {
        return;
    }
    else if(root -> left == nullptr && root -> right == nullptr)
    {
        return;
    }
    else if(encoded == "1")
    {
        decode(root -> right, encoded);
    }
    else
    {
        decode(root -> left, encoded);
    }
}
*/

void calc_histo(Mat image, Mat& hist)
{
    hist = Mat(256, 1, CV_32F, Scalar(0));

    for (int i = 0; i < image.rows;i++) {
        for (int j = 0; j < image.cols; j++) {
            int idx = int(image.at<uchar>(i, j));
            hist.at<float>(idx)++;
        }
    }

}


void build_huffman(Mat img, Mat hist)
{   
    int cnt = 0;
    int sum = 0;
    int frequency[256] = { 0 };
    for (int i = 0; i < 256; i++)
    {
        frequency[i] = hist.at<float>(i, 0);
    }

    //Min heap based on the comparison of the second pair
    //priority_queue<pair<char, int>, vector<pair<char, int>>, comparesecond > queue;
    priority_queue<node*, vector<node*>, comparesecond > queue;

    cout << "FREQUENCY OF INPUT IMAGES " << endl;
    for (int i = 0; i < 256; i++)
    {
            
            //i is the index of the ascii
            //queue.push(make_pair(char(i), frequency[i]));
            cout <<"Intensity:" <<int(i) << " frequency: " << frequency[i] << endl;
            
            if (frequency[i] != 0)
            {
                queue.push(new node(int(i), frequency[i]));
                cnt++;
            }
        
    }

    while (queue.size() > 1)
    {
        //Left has to be first as the smaller is the left leaf
        node* left = queue.top();
        queue.pop(); //Removes the top element. In this case, the lowest frequency. Doesn't return anything.
        node* right = queue.top();
        queue.pop();

        // IF TWO FREQUENCY ARE THE SAME, WE DO NOT KNOW THE ORDER (LEFT OR RIGHT)*/
        //WE SHOULD MAKE THE LEAST FREQUENT TO THE RIGHT*/


         //A smart algorithm is to then push the sum of the freuency into the priority queue
         //The summation of frequency will then sort itself out in the min heap
         //The frequency nodes will be connected if they are also one of the lowest frequency

        int sum = left->frequency + right->frequency;
        queue.push(new node(sum, left, right));//Since left is the smaller one, it is fine
    }
    //Top now contains a node with the total frequency
    node* root = queue.top();
    //Now transform the string in the tree to the shortened binary
    unordered_map <int, string> unorderedmap;
    //Traverse the tree and store the respective binary into the new tree
    string binary;
    encode(unorderedmap, root, binary);

    unordered_map<int, string>::iterator itr;
    cout << "\nAll Elements : \n";
        for (itr = unorderedmap.begin(); itr != unorderedmap.end(); itr++)
        {
            // itr works as a pointer to pair<string, double>
            // type itr->first stores the key part  and
            // itr->second stroes the value part
            cout << itr->first << "->" << itr->second << " ,code length: "<<itr->second.size()<<endl;
            
            
            sum += hist.at<float>(itr->first, 0) * itr->second.size();
            
        }

        float bits =1- float(sum)/ float((img.cols * img.rows * 8));
        printf("%d\n", sum);
        cout << fixed;
        cout.precision(8);
        cout << bits*100 << endl;

        
    
    //cout<< "encoded is " << result << endl;
    //2097152
}


void test()
{
        // priority_queue
        priority_queue< int, vector<int>, greater<int> > pq;

        // push(element)
        pq.push(5);
        pq.push(2);
        pq.push(8);
        pq.push(9);
        pq.push(1);
        pq.push(14);

        // pop()
        pq.pop();
        pq.pop();

        // top();
        cout << "pq top : " << pq.top() << '\n';

        // empty(), size()
        if (!pq.empty()) cout << "pq size : " << pq.size() << '\n';

        // pop all
        while (!pq.empty()) {
            cout << pq.top() << " ";
            pq.pop();
        }

        cout << '\n';
    
}

void erosion(Mat img, Mat& dst)
{
    uchar data[] = { 0,1,0,
                     1,1,1,
                     0,1,0 };
    
    Mat mask(3, 3, CV_8UC1, data);
    Mat temp = Mat::zeros(3, 3, CV_8UC1);
    Mat temp1 = Mat::zeros(3, 3, CV_8UC1);
    
    for (int i = 0; i < img.rows-2; i++)
    {
        for (int j = 0; j < img.cols-2; j++)
        {
            for (int u = 0; u < 3; u++)
            {
                for (int v = 0; v < 3; v++)
                {
                    temp.at<uchar>(u, v) = img.at<uchar>(i + u, j + v);

                }
            }
            temp1 = temp.mul(mask);

            if (sum(temp1) == Scalar(1275))
            {
                dst.at<uchar>(i, j) = 255;
            }
        }
    }
    
}

void Zero_padding(Mat img, Mat& dst, int Size)
{
    Point img_size = img.size();
    dst = Mat(img_size.y + 2 * Size, img_size.x + 2 * Size, CV_8UC1, Scalar(0));

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {

            int x = j;
            int y = i;
            dst.at<uchar>(y + Size, x + Size) = img.at<uchar>(i, j);
        }
    }

}

void Zero_padding_16(Mat img, Mat& dst, int Size)
{
    Point img_size = img.size();
    dst = Mat(img_size.y + 2 * Size, img_size.x + 2 * Size, CV_16UC1, Scalar(0));

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {

            int x = j;
            int y = i;
            dst.at<unsigned short int>(y + Size, x + Size) = img.at<unsigned short int>(i, j);
        }
    }

}

void Dilation(Mat image, Mat& dst, Mat mask)
{   
    Mat img;
    Zero_padding(image, img, 1);
    
    Mat temp = Mat::zeros(3, 3, CV_8UC1);
    Mat temp1 = Mat::zeros(3, 3, CV_8UC1);
    int mul;
    for (int i = 1; i < img.rows - 1; i++)
    {
        for (int j = 1; j < img.cols - 1; j++)
        {

            mul = img.at<uchar>(i - 1, j - 1) + img.at<uchar>(i - 1, j) + img.at<uchar>(i - 1, j + 1) + img.at<uchar>(i, j - 1) + img.at<uchar>(i, j) + img.at<uchar>(i, j + 1) + img.at<uchar>(i + 1, j - 1) + img.at<uchar>(i + 1, j) + img.at<uchar>(i + 1, j + 1);

            if (mul != 0)
            {
                dst.at<uchar>(i - 1, j - 1) = 255;
            }
        }
    }
    int k = 0;

}

bool check_match(Mat img, Point start, Mat mask, int mode = 0)
{
    for (int u = 0; u < mask.rows; u++)
    {
        for (int v = 0; v < mask.cols;v++)
        {
            Point pt(v, u);
            int m = mask.at<uchar>(pt);
            int p = img.at<uchar>(start + pt);

            bool ch = (p == 255);
            if (m == 1 && ch == mode)
                return false;
        }
    }
    return true;

}

void Erosion_cv(Mat img, Mat& dst, Mat mask)
{
    dst = Mat(img.size(), CV_8U, Scalar(0));
    if (mask.empty()) mask = Mat(3, 3, CV_8UC1, Scalar(1));

    Point h_m = mask.size() / 2;
    for (int i = h_m.y; i < img.rows - h_m.y; i++)
    {
        for (int j = h_m.x; j < img.cols - h_m.x; j++)
        {
            Point start = Point(j, i) - h_m;
            bool check = check_match(img, start, mask, 0);
            dst.at<uchar>(i, j) = (check) ? 255 : 0;


        }


    }

}

void Dilation_cv(Mat img, Mat& dst, Mat mask)
{
    dst = Mat(img.size(), CV_8U, Scalar(0));
    if (mask.empty()) mask = Mat(3, 3, CV_8UC1, Scalar(1));

    Point h_m = mask.size() / 2;
    for (int i = h_m.y; i < img.rows - h_m.y; i++)
    {
        for (int j = h_m.x; j < img.cols - h_m.x; j++)
        {
            Point start = Point(j, i) - h_m;
            bool check = check_match(img, start, mask, 0);
            dst.at<uchar>(i, j) = (check) ? 0: 255;

        }
    }
}


void D(Mat img, int x, int y, int m_x, int m_y, double& result, int S, int C)
{
    int D_coordinate;
    int D_intensity;
    int I_m;
    int I;
    //printf("%d,%d,%d,%d\n", x, y, m_x, m_y);
    D_coordinate = (x - m_x)*(x - m_x) +(y - m_y)*(y -m_y);
    //printf("%d\n", D_coordinate);

    I_m= (int)img.at<uchar>(m_x, m_y);
    I = (int)img.at<uchar>(x, y);

    D_intensity = (I_m - I)*(I_m - I)*3;
    //printf("%d,%d,%d\n", I_m,I,D_intensity);
     
    result = sqrt(D_intensity + (D_coordinate / (S ^ 2)) * (C ^ 2));
    //printf("%f\n", result);

}
void Superpixel(Mat img, int N, Mat& dst) {

    int S = img.cols / N;

    
    Mat NxN_D = Mat::zeros(S, S, CV_16U);
    Mat pad;

    Mat m_x = Mat::zeros(N, N, CV_16UC1);
    Mat m_y = Mat::zeros(N, N, CV_16UC1);
    
    if (S % 2 == 0)

        Zero_padding(img, pad, int(S / 2));

    else if(S % 2 == 1)

        Zero_padding(img, pad, int(S / 2) + 1);

    int u, v = 0;
    unsigned short int Uclid;
    double minVal, maxVal;
    Point minLoc, maxLoc;
    int minIdx[2] = {}, maxIdx[2] = {};
    for (int u = 0; u < N; u++)
    {
        for (int v = 0; v < N; v++)
        {
            int U = u * S ;
            int V = v * S ;

            for (int i = U+ S/2; i < U + S + (S / 2);i++)
            {
                for (int j = V + S/2; j < V + S + (S / 2); j++)
                {
                    int H_dif = pad.at<uchar>(i - 1, j) - pad.at<uchar>(i + 1, j);
                    int V_dif = pad.at<uchar>(i, j - 1) - pad.at<uchar>(i, j + 1);
                    Uclid = H_dif * H_dif + V_dif * V_dif;

                    //Uclid = (pad.at(i,j) - pad.at(i-1,j))(i - (S+U)) * (i - (S+U)) + (j - (S+V)) * (j - (S+V));
                    // printf("%d,%d,%d\n", i-U-S/2,j-V-S/2,Uclid);
                    // NxN_D.at<unsigned short int>(0, 0) = (unsigned short int)Uclid;
                    NxN_D.at<unsigned short int>((i- U-S/2), (j - V - S / 2)) = (unsigned short int)Uclid;
                }

            }
            minMaxIdx(NxN_D, &minVal, &maxVal,minIdx,maxIdx);
            minMaxLoc(NxN_D, 0, 0, &minLoc, &maxLoc);
            //printf("%d, %d, %d\n", minVal,minIdx[1],minIdx[0]);
            
            Point pt;
            int temp=100000;
            int Distance;
            for (int i = 0; i < NxN_D.rows; i++)
            {
                for (int j = 0; j < NxN_D.cols; j++)
                {
                    if (NxN_D.at<unsigned short int>(i,j) == (unsigned short int)minVal)
                    {

                        Distance = (i - S / 2) * (i - S / 2) + (j - S / 2) * (j - S / 2);

                        if (Distance < temp)
                        {
                            temp = Distance;
                            pt.x = i; 
                            pt.y = j;
                            //printf("%d,%d,%d,%d\n", i, j,Distance,NxN_D.at<unsigned short int>(i,j));
                        }

                    }

                }
            }
            //img.at<uchar>(pt.x+U,pt.y+V) = 0;
            m_x.at<unsigned short int>(u, v) = pt.x;
            m_y.at<unsigned short int>(u, v) = pt.y;
            //cout << pt.x << " " << pt.y <<" "<< endl;
            
        }
    }

    //imwrite("C:/Users/user/Desktop/DIP_Seminar/week4/pixel_center.tif", img);
 
    Mat pad2;
    Mat result;
    
    Mat img_label = Mat::zeros(img.rows, img.cols, CV_16UC1);
    Mat img_D = Mat::zeros(img.rows, img.cols, CV_16U);
    img_D = img_D + 1000000;
    int center_x;
    int center_y;
    int label= -1;
    int pixel = 0;
    Zero_padding(img, pad2,S);

    for (int u = 0; u < N; u++)
    {
        for (int v = 0; v < N; v++)
        {   
            int U = u * S;
            int V = v * S;
            center_x = m_x.at<unsigned short int>(u, v)+U;
            center_y = m_y.at<unsigned short int>(u, v)+V;
            //printf("%d,%d\n", center_x, center_y);
            label++;
            pixel = 0;
            for (int i = center_x;i < center_x + 2 * S; i++)
            {   
                for (int j = center_y;j < center_y + 2 * S; j++)
                {
                    double result;
                    D(pad2, i, j, center_x + S, center_y + S, result, S, 10);

                    if (i >= S && i <= img.rows+S-1  && j >= S && j <= img.cols+S-1 )
                    {

                        if (result < img_D.at<unsigned short int>(i - S, j - S))
                        {
                            pixel++;
                            img_D.at<unsigned short int>(i - S, j - S) = result;
                            img_label.at<unsigned short int>(i - S, j - S) = label;
                            //printf("pixel num: %d, label num : %d\n", pixel, label);
                        }
                    }
                }
            }

        }
    }
    label = -1;
    int x_sum=0;
    int y_sum=0;
    int cnt = 0;
    int mean_x = 0;
    int mean_y = 0;
    int mean_val = 0;
    Mat E_converge = Mat::zeros(N, N, CV_8UC1);
    float E;
    Mat temp_x = m_x.clone();
    Mat temp_y = m_y.clone();
    Mat img_cvt = img.clone();
    for (int u = 0; u < N; u++)
    {
        for (int v = 0; v < N; v++)
        {
            int U = u * S;
            int V = v * S;
            center_x = m_x.at<unsigned short int>(u, v) + U;
            center_y = m_y.at<unsigned short int>(u, v) + V;
            label++;
            for (int i = center_x;i < center_x + 2 * S; i++)
            {
                for (int j = center_y;j < center_y + 2 * S; j++)
                {
                    if (i >= S && i <= img.rows + S - 1 && j >= S && j <= img.cols + S - 1)
                    {

                        if (img_label.at<unsigned short int>(i - S, j - S)==label)
                        {
                            cnt++;
                            x_sum += i - S;
                            y_sum += j - S;
                            //printf("pixel num: %d, label num : %d\n", pixel, label);
                        }
                    }
                }
            }
          
            
           
                mean_x = ceil(x_sum / cnt);
                mean_y = ceil(y_sum / cnt);

                mean_val = img.at<uchar>(mean_x, mean_y);

                for (int i = center_x;i < center_x + 2 * S; i++)
                {
                    for (int j = center_y;j < center_y + 2 * S; j++)
                    {
                        if (i >= S && i <= img.rows + S - 1 && j >= S && j <= img.cols + S - 1)
                        {

                            if (img_label.at<unsigned short int>(i - S, j - S) == label)
                            {
                                img_cvt.at<uchar>(i - S, j - S) = mean_val;
                                //printf("pixel num: %d, label num : %d\n", pixel, label);
                            }
                        }
                    }
                }

            
                cnt = 0;
                x_sum = 0;
                y_sum = 0;
                E = sqrt((mean_x - center_x) * (mean_x - center_x) + (mean_y - center_y) * (mean_y - center_y));

                if (mean_x - U >= 0)
                    m_x.at<unsigned short int>(u, v) = mean_x - U;
                else if (mean_x - U < 0)
                    m_x.at<unsigned short int>(u, v) = 0;

                if (mean_y - V >= 0)
                    m_y.at<unsigned short int>(u, v) = mean_y - V;
                else if (mean_y - V < 0)
                    m_y.at<unsigned short int>(u, v) = 0;
                //temp_x.at<unsigned short int>(u, v) = mean_x ;
                //temp_y.at<unsigned short int>(u, v) = mean_y ;
                E_converge.at<uchar>(u, v) = E;          
        }

    }
    /*
    img_label.convertTo(img_label, CV_8UC1);
    Mat img_label_clone = img_label.clone();
    for (int i = 1; i < img_label.rows-1; i++)
        for (int j = 1; j < img_label.cols-1; j++)
        {
            if (img_label.at<uchar>(i - 1, j) == img_label.at<uchar>(i, j - 1) == img_label.at<uchar>(i + 1, j) == img_label.at<uchar>(i, j + 1) != img_label.at<uchar>(i, j))
            {
                img_label_clone.at<uchar>(i, j) = img_label.at<uchar>(i - 1, j);
            }
        }

    img_label = img_label_clone;
    */

    //Need to determine CV_8UC1 or CV_16UC1 by superpixel size

    dst = img_cvt;
    Mat img_label_view;
    img_label.convertTo(img_label_view, CV_16UC1);
    //imwrite("C:/Users/user/Desktop/DIP_Seminar/week4/img_label_32.tif",img_label_view);
    Mat label_test;
    Mat pad11;

    Zero_padding_16(img_label_view, label_test,1);
    Zero_padding(img, pad11, 1);
    pad11.convertTo(pad11, CV_16UC1);

    label_test.convertTo(label_test, CV_16UC1);

    Mat label_view = Mat::zeros(label_test.rows, label_test.cols, CV_16UC1);
    label_view += 128;
    for(int i =1; i <label_test.rows-1; i++)
        for (int j = 1; j < label_test.cols-1; j++)
        {  
            if (label_test.at<unsigned short int>(i - 1, j) + label_test.at<unsigned short int>(i, j - 1) + label_test.at<unsigned short int>(i + 1, j) + label_test.at<unsigned short int>(i, j + 1) == 4 * label_test.at<unsigned short int>(i, j))
                label_view.at<unsigned short int>(i, j) = pad11.at<unsigned short int>(i, j);
        }
    label_view.convertTo(label_view, CV_8UC1);
    imwrite("C:/Users/user/Desktop/DIP_Seminar/week4/label_test_test_64.tif", label_view);

}


void holefilling(Mat img, Mat& dst, int iteration)
{
    Mat I = img;
    
    Mat I_c= Mat::zeros(img.rows, img.cols, CV_8UC1);
    Mat F = Mat::zeros(img.rows, img.cols, CV_8UC1);
    Mat H = Mat::zeros(img.rows, img.cols, CV_8UC1);
    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++)
        {
            if (img.at<uchar>(i, j) == 255)
                I_c.at<uchar>(i, j) = 0;
            else
                I_c.at<uchar>(i, j) = 255;
            
            if (i == img.rows-1 || j == img.cols-1 || i == 0 || j == 0)
                F.at<uchar>(i, j) = 255 - I.at<uchar>(i, j);
        
        }
    imwrite("C:/Users/user/Desktop/DIP_Seminar/week4/F.tif", F);
    imwrite("C:/Users/user/Desktop/DIP_Seminar/week4/I_c.tif", I_c);

    uchar data[] = { 1,1,1,
                 1,1,1,
                 1,1,1 };

    Mat mask(3, 3, CV_8UC1, data);

    for (int k = 0; k < iteration; k++)
    {
        Dilation(F, F, mask);
        F = F.mul(I_c);
        printf("%d iteration\n",k);
        if(k ==29)
            imwrite("C:/Users/user/Desktop/DIP_Seminar/week4/29.tif", F);
        else if(k==100)
            imwrite("C:/Users/user/Desktop/DIP_Seminar/week3/100.tif", F);
        else if(k==300)
            imwrite("C:/Users/user/Desktop/DIP_Seminar/week3/300.tif", F);
        else if(k==500)
            imwrite("C:/Users/user/Desktop/DIP_Seminar/week3/500.tif", F);
        else if(k==700)
            imwrite("C:/Users/user/Desktop/DIP_Seminar/week3/700.tif", F);
        else if(k ==900)
           imwrite("C:/Users/user/Desktop/DIP_Seminar/week3/700.tif", F);

    }
    dst = F;
    /*
    for (int i = 0; i < H.rows; i++)
        for (int j = 0; j < H.cols; j++)
        {
            if (H.at<uchar>(i, j) == 255)
                H.at<uchar>(i, j) = 0;
            else
                H.at<uchar>(i, j) = 255;
        }
     */
    
}

int main()
{
    

    //Mat img = imread("C:/Users/user/Desktop/DIP_Seminar/week4/wirebond.tif", IMREAD_GRAYSCALE);
    Mat img2 = imread("C:/Users/user/Desktop/DIP_Seminar/week4/Lenna_gray.tif", IMREAD_GRAYSCALE);
    
    //Superpixel
    Mat superpixel64;
    Mat superpixel32;
    Mat superpixel16;
    Superpixel(img2, 64, superpixel16);
    imwrite("C:/Users/user/Desktop/DIP_Seminar/week4/superpixel64_fix.tif", superpixel16);

    

    /*
    Superpixel(img2, 32, superpixel32);
    imwrite("C:/Users/user/Desktop/DIP_Seminar/week4/superpixel32_test.tif", superpixel32);

    Superpixel(img2, 64, superpixel64);
    imwrite("C:/Users/user/Desktop/DIP_Seminar/week4/superpixel64_test.tif", superpixel64);

    */

    //Mat result = Mat::zeros(img.rows-2, img.cols-2, CV_8UC1);
    //Dilation(img, result);

    /*//Hole filling
    Mat padded;
    Mat result = Mat::zeros(img2.rows, img2.cols, CV_8UC1);
   
    uchar data[] = { 1,1,1,
                     1,1,1,
                     1,1,1 };

    Mat mask(3, 3, CV_8UC1, data);
    
    holefilling(img2,result,5);
    
    for (int i = 0; i < result.rows; i++)
        for (int j = 0; j < result.cols; j++)
        {
            if (result.at<uchar>(i, j)==255)
                result.at<uchar>(i, j) = 0;
            else
                result.at<uchar>(i, j) = 255;
        }
    imshow("img", result);
     waitKey(0);
    imwrite("C:/Users/user/Desktop/DIP_Seminar/week4/result_t.tif" , result);
    */

    //*/

    /*Huffman coding
    Mat hist;
    calc_histo(img2, hist);

    //cout << hist.t() << endl;
    build_huffman(img2,hist);
    */



    /*
    int i, j;
    int z;
    ofstream output;
    output.open("file.txt");
    */

    /*
    for (i = 0; i < img.rows;i++)
    {
        for (j = 0; j < img.cols; j++)
        {
            printf("%d\n", img.at<uchar>(i, j));
            
        }
    }
    */

    /*
    if (output.is_open())
    {
        for(i =0; i<img.rows;i++)
        {
            for (j = 0; j < img.cols; j++)
            {
                output << (int)img.at<uchar>(i, j);
                output << "\n";
            }
        }
    }
    output.close();
    */

    /*
    ifstream test;
    test.open("file.txt");
    
    int st[100];

    if (test.is_open())
    {
        for(i=0; i<100; i++)
        {
            test >> st[cnt];
            cnt++;
        }
    }

    else
    {
        cout << "file error";
    }

    i = 0;
    while (i < cnt - 1) {

        cout << st[i] << "\n";
        i++;
    }
    
    cout << "Done " << endl;
    return 0;
    
    
    */


    /*
    ifstream test1;
    test1.open("file.txt");
    string test = get_input(test1);
    
    build_huffman(test, test.size());
    
    return 1;
    */
}