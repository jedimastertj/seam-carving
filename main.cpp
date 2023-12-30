#include "bits/stdc++.h"
#include "opencv4/opencv2/opencv.hpp"
#include "unistd.h"
#include "sys/wait.h"
#include "fstream"

using namespace std;
using namespace cv;

typedef pair<int, int> pi;
int r, c;
int vsr, hsr;
int removeEnergy = -10000; // incentivize removal of pixel

// returns intensity of pixel given rgb values
int getIntensity(Vec3b& bgr) {
    return (0.3 * bgr[2] + 0.59 * bgr[1] + 0.11 * bgr[0]);
}

// returns energy of pixel as the 2-norm of intensity gradient
int getEnergy(vector<vector<pair<int, pi>>>& intensity, int row, int col, bool customSize) {
    int rowS = r-hsr, colS = c-vsr;
    if (customSize) {
        rowS = intensity.size();
        colS = intensity[0].size();
    }
    
    int left = (col == 0) ? 0 : intensity[row][col-1].first;
    int right = (col == colS-1) ? 0 : intensity[row][col+1].first;
    int up = (row == 0) ? 0 : intensity[row-1][col].first;
    int down = (row == rowS-1) ? 0 : intensity[row+1][col].first;

    return sqrt(pow(right - left, 2) + pow(down - up, 2));
}

// updates energy matrix based on intensity matrix 
void setEnergy(vector<vector<pair<int, pi>>>& intensity, vector<vector<int>>& energy, bool customSize) {
    int rowS = r-hsr, colS = c-vsr;
    if (customSize) {
        rowS = intensity.size();
        colS = intensity[0].size();
    }

    // resize to current dimensions
    energy.resize(rowS);
    for (int i = 0; i < rowS; i++) energy[i].resize(colS, -1);

    for (int row = 0; row < rowS; row++) {
        for (int col = 0; col < colS; col++) {
            energy[row][col] = getEnergy(intensity, row, col, customSize);
        }
    } 
}

// updates cost matrix based on energy matrix and type
void setCost(vector<vector<int>>& energy, vector<vector<int>>& cost, int type, bool customSize) {
    int rowS = r-hsr, colS = c-vsr;
    if (customSize) {
        rowS = energy.size();
        colS = energy[0].size();
    }
    
    // resize to current dimensions
    cost.resize(rowS);
    for (int i = 0; i < r-hsr; i++) cost[i].resize(colS, -1);

    // for vertical seam operations
    if (type == 0) { 
        for (int row = 0; row < rowS; row++) {
            for (int col = 0; col < colS; col++) {
                if (row == 0) cost[row][col] = energy[row][col];
                else {
                    int prev = cost[row-1][col];
                    if (col > 0) prev = min(prev, cost[row-1][col-1]);
                    if (col < c-1-vsr) prev = min(prev, cost[row-1][col+1]);
                    cost[row][col] = energy[row][col] + prev;
                }
            }
        }
    }
    // for horizontal seam operations
    else if (type == 1) { 
        for (int col = 0; col < colS; col++) {
            for (int row = 0; row < rowS; row++) {
                if (col == 0) cost[row][col] = energy[row][col];
                else {
                    int prev = cost[row][col-1];
                    if (row > 0) prev = min(prev, cost[row-1][col-1]);
                    if (row < r-1-hsr) prev = min(prev, cost[row+1][col-1]);
                    cost[row][col] = energy[row][col] + prev;
                }
            }
        }
    }
}

// computes optimal seam based on cost matrix and type
void getOptimalSeam(vector<vector<int>>& cost, int type, vector<pi>& seam, bool customSize) {
    int rowS = r-hsr, colS = c-vsr;
    if (customSize) {
        rowS = cost.size();
        colS = cost[0].size();
    }

    seam.clear(); // avoid errors
    int val = INT_MAX, row = -1, col = -1;

    if (type == 0) { // vertical seam
        row = rowS-1;
        
        for (int j = 0; j < colS; j++) {
            if (cost[row][j] < val) {
                col = j;
                val = cost[row][j];
            }
        }
        seam.push_back({row, col});
        
        while (--row >= 0) {
            int col_temp = col;
            val = cost[row][col];

            if (col > 0 && cost[row][col-1] < val) {
                val = cost[row][col-1];
                col_temp = col-1;
            }
            if (col < c-1-vsr && cost[row][col+1] < val) {
                val = cost[row][col+1];
                col_temp = col+1;
            }
            
            col = col_temp;
            seam.push_back({row, col});
        }
    }
    else if (type == 1) { // horizontal seam
        col = colS-1;

        for (int i = 0; i < rowS; i++) {
            if (cost[i][col] < val) {
                row = i;
                val = cost[i][col];
            }
        }
        seam.push_back({row, col});

        while (--col >= 0) {
            int row_temp = row;
            val = cost[row][col];

            if (row > 0 && cost[row-1][col] < val) {
                val = cost[row-1][col];
                row_temp = row-1;
            }
            if (row < r-1-hsr && cost[row+1][col] < val) {
                val = cost[row+1][col];
                row_temp = row+1;
            }
            
            row = row_temp;
            seam.push_back({row, col});
        }
    }
}

// updates intensity matrix to swap out removed seam of pixels 
void swapRemoved(vector<vector<pair<int, pi>>>& intensity, int type, bool customSize) {    
    int rowS = r-hsr, colS = c-vsr;
    if (customSize) {
        rowS = intensity.size();
        colS = intensity[0].size();
    }
    
    // after vertical seam removal
    if (type == 0) { 
        for (int row = 0; row < rowS; row++) {
            for (int col = 0; col < colS-1; col++) {
                if (intensity[row][col].first == -1) swap(intensity[row][col], intensity[row][col+1]);
            }
        }
    } 
    // after horizontal seam removal
    else if (type == 1) {
        for (int col = 0; col < colS; col++) {
            for (int row = 0; row < rowS-1; row++) {
                if (intensity[row][col].first == -1) swap(intensity[row][col], intensity[row+1][col]);
            }
        }
    } 
}

// removes vertical/horizontal seam
void removeSeam(vector<vector<int>>& cost, vector<vector<pair<int, pi>>>& intensity, int type, bool customSize) {
    int rowS = r-hsr, colS = c-vsr;
    if (customSize) {
        rowS = intensity.size();
        colS = intensity[0].size();
    }
    
    vector<pi> seam;
    getOptimalSeam(cost, type, seam, customSize);
    for (pi pixel: seam) intensity[pixel.first][pixel.second].first = -1;
    swapRemoved(intensity, type, customSize);

    if (type == 0) colS -= 1;
    else rowS -= 1;

    intensity.resize(rowS);
    for (int i = 0; i < rowS; i++) intensity[i].resize(colS);

    if (!customSize) {
        if (type == 0) vsr++;
        else hsr++;
    }
}

Vec3b averageColor(Vec3b colorA, Vec3b colorB) {
    return Vec3b((colorA[0]+colorB[0])/2, (colorA[1]+colorB[1])/2, (colorA[2]+colorB[2])/2);
}

// return color value for pixel at row i, col j as per current intensity matrix
// either pixel is present in original image, or it was added - hence compute color recursively (as need may arise)

Vec3b getColor(vector<vector<pair<int, pi>>>& intensity, Mat& image, int i, int j) {
    // handle cases of going out of boundaries
    if (i < 0 || i >= r-hsr) return Vec3b(0, 0, 0);
    if (j < 0 || j >= c-vsr) return Vec3b(0, 0, 0);

    pi original = intensity[i][j].second;

    if (original.first == -1 && original.second == -1) {
        // horizontal merge pixel
        Vec3b colorA = getColor(intensity, image, i, j-1); // left
        int tracker = j+1;
        while (tracker < c-vsr && intensity[i][tracker].second.first < 0) tracker++; 

        Vec3b colorB = getColor(intensity, image, i, tracker); // right
        return averageColor(colorA, colorB);

    } else if (original.first == -2 && original.second == -2) {
        // vertical merge pixel
        Vec3b colorA = getColor(intensity, image, i-1, j); // up
        int tracker = i+1;
        while (tracker < r-hsr && intensity[tracker][j].second.first < 0) tracker++;

        Vec3b colorB = getColor(intensity, image, tracker, j); // down
        return averageColor(colorA, colorB);

    } else if (original.first >= 0 && original.first < image.rows && original.second >= 0 && original.second < image.cols) {
        // simple pixel - present in original image
        return image.at<Vec3b>(original.first, original.second);
    } 
    else return Vec3b(0, 0, 0);
}

void insertSeams(vector<vector<pair<int, pi>>>& intensity, vector<vector<int>>& cost, vector<vector<int>>& energy, Mat& image, int type, int k) {

    vector<vector<pair<int, pi>>> intensity_temp(r-hsr, vector<pair<int, pi>>(c-vsr));

    for (int i = 0; i < r-hsr; i++) {
        for (int j = 0; j < c-vsr; j++) {
            intensity_temp[i][j].first = intensity[i][j].first;
            intensity_temp[i][j].second = {i, j}; // *** instead of coordinates in original image, store coordinates in initial intensity matrix to support insertion of seams 
        }
    }

    vector<vector<bool>> toCopy(r-hsr, vector<bool>(c-vsr, false)); // marks which pixels from initial intensity matrix being removed and hence will be added 

    // limit for enlarging in one go
    if (type == 0) k = min(k, (c-vsr)/5);
    else if (type == 1) k = min(k, (r-hsr)/5);

    // simulate seam removal on temp matrix
    for (int i = 0; i < k; i++) {
        setEnergy(intensity_temp, energy, false);
        setCost(energy, cost, type, false);

        vector<pi> seam;
        getOptimalSeam(cost, type, seam, false);
        for (pi pixel: seam) {
            pi initial = intensity_temp[pixel.first][pixel.second].second;

            toCopy[initial.first][initial.second] = true;
            intensity_temp[pixel.first][pixel.second].first = -1;
        }

        swapRemoved(intensity_temp, type, false);
        
        if (type == 0) vsr++;
        else if (type == 1) hsr++;
    }

    // reset
    if (type == 0) vsr -= k; 
    else if (type == 1) hsr -= k;

    vector<vector<pair<int, pi>>> intensity_final(r-hsr);

    if (type == 0) {
        for (int i = 0; i < r-hsr; i++) {
            for (int j = 0; j < c-vsr; j++) {
                intensity_final[i].push_back(intensity[i][j]);
                if (toCopy[i][j]) {
                    // insert horizontal merge pixel "on right"
                    // we will fill correct value of intensity (by computing color) after completing structure of intensity_final (placing all pixels)
                    intensity_final[i].push_back({-1, {-1, -1}}); // {-1, -1} denotes horizontal merge pixel
                }
            }
        }
        vsr -= k;
    }
    else if (type == 1) {
        intensity_final.resize(r-hsr+k);
        for (int i = 0; i < r-hsr+k; i++) intensity_final[i].resize(c-vsr);
        
        for (int j = 0; j < c-vsr; j++) {
            int skip = 0;
            for (int i = 0; i < r-hsr; i++) {
                intensity_final[i+skip][j] = intensity[i][j];
                if (toCopy[i][j]) {
                    // insert vertical merge pixel "below"
                    // we will fill correct value of intensity (by computing color) after completing structure of intensity_final (placing all pixels)
                    skip++;
                    intensity_final[i+skip][j] = {-1, {-2, -2}}; // {-2, -2} denotes vertical merge pixel
                }
            }
        }
        hsr -= k;
    }

    for (int i = 0; i < r-hsr; i++) {
        for (int j = 0; j < c-vsr; j++) {
            if (intensity_final[i][j].first == -1) {
                // set intensity
                Vec3b computedColor = getColor(intensity_final, image, i, j);
                intensity_final[i][j].first = getIntensity(computedColor);
            }
        }
    }

    // final setup
    intensity.resize(r-hsr);
    for (int i = 0; i < r-hsr; i++) intensity[i].resize(c-vsr);

    for (int i = 0; i < r-hsr; i++) {
        for (int j = 0; j < c-vsr; j++) {
            intensity[i][j] = intensity_final[i][j];
        }
    }
}

// sets special energy for marked pixels in object removal application
void tweakEnergy(vector<vector<pair<int, pi>>>& intensity, vector<vector<int>>& energy, vector<vector<bool>>& remove) {
    for (int i = 0; i < energy.size(); i++) {
        for (int j = 0; j < energy[0].size(); j++) {
            pi original = intensity[i][j].second;
            if (remove[original.first][original.second]) energy[i][j] = removeEnergy;
        }
    }
}

int main(int argc, char* argv[]) {
    
    // image setup

    string filename, outName;
    int fr, fc;
    float scale;
    bool retargeting = false, optimalRetargeting = false, contentAmplify = false, objectRemoval = false, multiSize = false;

    int choice;
    cout << "Seam carving and its applications: " << endl;
    cout << "1. Simple image retargeting" << endl;
    cout << "2. Optimal image retargeting" << endl;
    cout << "3. Content amplification" << endl;
    cout << "4. Multi-size image" << endl;
    cout << "5. Object removal" << endl;
    cout << "Enter choice number: ";
    cin >> choice;
    
    if (choice == 1) retargeting = true;
    else if (choice == 2) optimalRetargeting = true;
    else if (choice == 3) contentAmplify = true;
    else if (choice == 4) multiSize = true;
    else if (choice == 5) objectRemoval = true;
    else return 0;

    cout << "Enter name of input file (with ext): ";
    cin >> filename;
    
    if (retargeting || optimalRetargeting) {
        cout << "Enter output width and height (space separated): ";
        cin >> fc >> fr;
        if (fr <= 0 || fc <= 0) {
            cout << "invalid value for output height/width \n";
            return -1;
        }
    } 
    
    if (!multiSize) {
        cout << "Enter name of output file (with ext): ";
        cin >> outName;
    }

    if (multiSize) {
        cout << "Enter maximum supported width and height (space separated): ";
        cin >> fc >> fr;
        if (fr <= 0 || fc <= 0) {
            cout << "invalid value for output height/width \n";
            return -1;
        }

        int len = filename.size();
        string maxName = filename.substr(0, len-4) + "-max" + filename.substr(len-4, 4);

        int pid = fork();
        
        if (pid == 0) { // child
            multiSize = false;

            outName = maxName;
            retargeting = true;
        }
        else if (pid > 0) { // parent
            wait(NULL);

            Mat maxImage = imread(maxName, IMREAD_COLOR);
            r = maxImage.rows, c = maxImage.cols;
            vsr = 0, hsr = 0;

            vector<vector<pair<int, pi>>> intensity(r, vector<pair<int, pi>>(c));
            vector<vector<int>> energy(r, vector<int>(c));
            vector<vector<int>> cost(r, vector<int>(c));

            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    intensity[i][j].first = getIntensity(maxImage.at<Vec3b>(i, j));
                    intensity[i][j].second = {i, j};
                }
            }

            vector<vector<int>> removingSeam(r, vector<int>(c, INT_MAX));

            int scDir;
            cout << "Enter seam carving direction (v-0, h-1): ";
            cin >> scDir;

            if (scDir == 0) fc = 0;
            else fr = 0;
            
            int count = 0;
            while (r - hsr > fr || c - vsr > fc) {
                setEnergy(intensity, energy, false);
                setCost(energy, cost, scDir, false);
                
                int rowS = r-hsr, colS = c-vsr;
                vector<pi> seam;
                getOptimalSeam(cost, scDir, seam, false);
                ++count;

                for (pi pixel: seam) {
                    intensity[pixel.first][pixel.second].first = -1;
                    pi x = intensity[pixel.first][pixel.second].second;
                    removingSeam[x.first][x.second] = count;
                }

                swapRemoved(intensity, scDir, false);

                if (scDir == 0) colS -= 1;
                else rowS -= 1;

                intensity.resize(rowS);
                for (int i = 0; i < rowS; i++) intensity[i].resize(colS);

                if (scDir == 0) vsr++;
                else hsr++;
            }

            fstream outFile(maxName.substr(0, len) + ".txt", ios_base::out);
            outFile << scDir << "\n";
            for (auto row: removingSeam) {
                for (int val: row) outFile << val << " ";
                outFile << "\n"; 
            }  

            return 0;
        }
    }

    Mat image = imread(filename, IMREAD_COLOR);
    
    if (image.empty()) {
        cout << "couldn't read input image \n";
        return -1;
    } 
    else {
        if (contentAmplify) {
            fr = image.rows, fc = image.cols;
            cout << "Enter scaling factor: ";
            cin >> scale;

            int sr = fr * scale, sc = fc * scale;
            Mat scaledImage(sr, sc, CV_8UC3);

            for (int i = 0; i < sr; i++) {
                for (int j = 0; j < sc; j++) {
                    scaledImage.at<Vec3b>(i, j) = image.at<Vec3b>((int) i/scale, (int) j/scale);
                }
            }
            image = scaledImage;
        } 

        // number of rows, columns in original image 
        r = image.rows, c = image.cols;
        
        // vertical/horizontal seams removed (will be -ve when inserting seams)
        vsr = 0, hsr = 0; 

        // stores intensity value and co-ordinates of pixel in original image 
        vector<vector<pair<int, pi>>> intensity(r, vector<pair<int, pi>>(c));
        
        // stores energy value for each pixel
        vector<vector<int>> energy(r, vector<int>(c));
        
        // stores values of minimum cumulative energy out of all seams arriving from top/left
        vector<vector<int>> cost(r, vector<int>(c));

        // initialize intensity matrix
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                intensity[i][j].first = getIntensity(image.at<Vec3b>(i, j));
                intensity[i][j].second = {i, j};
            }
        }

        // stores whether to remove pixel
        vector<vector<bool>> remove(r, vector<bool>(c, false));

        int lm = image.cols, tm = image.rows, rm = 0, bm = 0;
        bool found = false;

        if (objectRemoval) {
            fr = r;
            fc = c;

            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    Vec3b color = image.at<Vec3b>(i, j);
                    // color is encoded as bgr values
                    if (color[2] >= 252 && color[0] <= 2 && color[1] <= 2) {
                        found = true;
                        remove[i][j] = true;
                        if (i < tm) tm = i;
                        if (i > bm) bm = i;
                        if (j < lm) lm = j;
                        if (j > rm) rm = j;
                    }
                }
            }

            if (found) {
                int dir;
                cout << "Enter direction of seam removal (v-0, h-1): ";
                cin >> dir;
                
                if (dir == 0) fc -= (rm-lm);
                else fr -= (bm-tm);
            } 
        }

        if (optimalRetargeting) {
            if (fc > c || fr > r) return 0;

            // from a given state, move = 0 means up and move = 1 means left
            vector<vector<int>> optimalMoveFrom(r-fr, vector<int>(c-fc, -1));

            // dp state includes optimalCost and corresponding resulting intensity matrix up till given cell
            // at any instant, need dp states for atleast all cells of previous row
            // keep updating row when processing cells of next row
            vector<pair<int, vector<vector<pair<int, pi>>>>> dp(r-fr);
            
            // base case - original image and cost
            dp[0] = {0, intensity}; 

            for (int j = 0; j < c-fc; j++) {
                for (int i = 0; i < r-fr; i++) {
                    if (i == 0 && j == 0) continue;
                    vector<vector<pair<int, pi>>> intensity_copy;

                    int optimalCost = INT_MAX;
                    vector<vector<pair<int, pi>>> intensity_afterwards;
                    
                    // up
                    if (j > 0) {
                        int prevCost = dp[i].first;
                        intensity_copy = dp[i].second;
                        
                        // vertical seam removal
                        setEnergy(intensity_copy, energy, true);
                        setCost(energy, cost, 0, true);

                        vector<pi> seam;
                        getOptimalSeam(cost, 0, seam, true);
                        int seamCost = cost[seam[0].first][seam[0].second];

                        if (prevCost + seamCost < optimalCost) {
                            optimalMoveFrom[i][j] = 0;
                            optimalCost = prevCost + seamCost;

                            for (pi pixel: seam) intensity_copy[pixel.first][pixel.second].first = -1;
                            swapRemoved(intensity_copy, 0, true);
                            intensity_afterwards = intensity_copy;
                        }
                    }

                    // left
                    if (i > 0) {
                        int prevCost = dp[i-1].first;
                        intensity_copy = dp[i-1].second;

                        // horizontal seam removal
                        setEnergy(intensity_copy, energy, true);
                        setCost(energy, cost, 1, true);

                        vector<pi> seam;
                        getOptimalSeam(cost, 1, seam, true);
                        int seamCost = cost[seam[0].first][seam[0].second];

                        if (prevCost + seamCost < optimalCost) {
                            optimalMoveFrom[i][j] = 1;
                            optimalCost = prevCost + seamCost;

                            for (pi pixel: seam) intensity_copy[pixel.first][pixel.second].first = -1;
                            swapRemoved(intensity_copy, 1, true);
                            intensity_afterwards = intensity_copy;
                        }
                    }

                    dp[i] = {optimalCost, intensity_afterwards};
                }
            }

            pi current = {r-fr-1, c-fc-1};
            stack<int> optimalOps; // 0 denotes vertical seam removal, 1 denotes horizontal seam removal
            while (optimalMoveFrom[current.first][current.second] != -1) {
                optimalOps.push(optimalMoveFrom[current.first][current.second]);

                if (optimalOps.top() == 0) current.second -= 1;
                else current.first -= 1;
            }

            while (!optimalOps.empty()) {
                int type = optimalOps.top();

                setEnergy(intensity, energy, false);
                setCost(energy, cost, type, false);
                removeSeam(cost, intensity, type, false);

                optimalOps.pop();
            }
        } 
        else {
            // while current columns > output columns
            while (c - vsr > fc) {
                // remove vertical seams
                setEnergy(intensity, energy, false);
                if (found) tweakEnergy(intensity, energy, remove);
                setCost(energy, cost, 0, false);
                removeSeam(cost, intensity, 0, false);
            }

            // while current rows > output rows
            while (r - hsr > fr) {
                // remove horizontal seams
                setEnergy(intensity, energy, false);
                if (found) tweakEnergy(intensity, energy, remove);
                setCost(energy, cost, 1, false);
                removeSeam(cost, intensity, 1, false);
            }    

            // make up for removed seams in object removal 
            if (found) {
                fr = r;
                fc = c;
            }

            // while current columns < output columns
            while (c - vsr < fc) {
                // insert vertical seams
                insertSeams(intensity, cost, energy, image, 0, fc-(c-vsr));
            }

            // while current rows < output rows
            while (r - hsr < fr) {
                // insert horizontal seams
                insertSeams(intensity, cost, energy, image, 1, fr-(r-hsr));
            }
        }

        // output final image 
        Mat output(fr, fc, CV_8UC3);
        for (int i = 0; i < fr; i++) {
            for (int j = 0; j < fc; j++) {
                output.at<Vec3b>(i, j) = getColor(intensity, image, i, j);
            }
        }
        imwrite(outName, output);
    } 

    return 0;
}
