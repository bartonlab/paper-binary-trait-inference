#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cassert>

void get_matrix(FILE *, std::vector<std::vector<double> > &);
void row_reduce(std::vector<std::vector<double> > &);

int main()
{
    // read in A entries in scientific format from a file cov.np.dat
    std::vector<std::vector<double> > A;
    
    if (FILE *datain = fopen("temp_cov.np.dat", "r")) {
    
        get_matrix(datain, A);
        fclose(datain);
        
    } else {
    
        std::cout << "rref.cpp - main: could not open file cov.np.dat\n" << std::endl;
        return EXIT_FAILURE;

    }

    row_reduce(A);

    // save the A to a file rref.np.dat
    std::ofstream out("temp_rref.np.dat");
    for (int i=0; i<A.size(); ++i) {
        for (int j=0; j<A[i].size(); ++j)
            out << A[i][j] << '\t';
        out << "\n";
    }

    return EXIT_SUCCESS;

}

double calculateEpsilon(const std::vector<std::vector<double> > &A) {
    double maxValue = 0.0;
    for (const auto &row : A) {
        for (double value : row) {
            maxValue = std::max(maxValue, fabs(value));
        }
    }
    return maxValue * 5e-8;
}

// double calculateEpsilon(const std::vector<std::vector<double> > &A) {
//     double minValue = 0.0; 
//     for (const auto &row : A) {
//         for (double value : row) {
//             if (value != 0.0) {
//                 minValue = std::min(minValue, fabs(value));
//             }
//         }
//     }

//     return minValue/10;
// }

// void row_reduce(std::vector<std::vector<double> > &A) {
//     int lead = 0;
//     // double EPSILON = calculateEpsilon(A); // set EPSILON
//     double EPSILON = 0; // set EPSILON

//     for (int row=0; row<A.size(); ++row) {
//         if (lead>A[row].size()) return;

//         int i = row;
        
//         // Find the row with a non-zero leading value
//         while (fabs(A[i][lead]) < EPSILON) {
//             ++i;
//             if (i>A.size()-1) {
//                 i = row;
//                 ++lead;
//                 if (lead>A[i].size()-1) return;
//             }
//         }
        
//         // Swap the current row with the row having the non-zero leading value
//         std::swap(A[i], A[row]);

//         // Normalize the leading value to 1
//         double div = A[row][lead];
//         if (fabs(div) > EPSILON) {
//             for (int j=0; j<A[row].size(); ++j) {
//                 A[row][j] /= div;
//                 if (fabs(A[row][j]) < EPSILON) A[row][j] = 0.0; // Ensure zero is zero
//             }
//         }

//         // Eliminate other entries in the leading column
//         for (i=0; i<A.size(); ++i) {
//             if (i!=row) {
//                 double mult = -A[i][lead];
//                 for (int j = 0; j < A[i].size(); ++j) {
//                     A[i][j] += mult * A[row][j];
//                     if (fabs(A[i][j]) < EPSILON) A[i][j] = 0.0; // Ensure zero is zero
//                 }
//             }
//         }
//         ++lead;
//     }

//     // Post-processing: Ensure zero rows are properly handled
//     for (int i = 0; i < A.size(); i++) {
//         bool zeroRow = true; // Track if the row is a zero row
//         for (int j = 0; j < A[i].size(); j++) {
//             if (fabs(A[i][j]) > EPSILON) {
//                 zeroRow = false;
//                 break;
//             }
//         }
//         if (zeroRow) {
//             for (int j = 0; j < A[i].size(); j++) {
//                 A[i][j] = 0.0; // set this row to zero row
//             }
//         }
//     }
// }

void row_reduce(std::vector<std::vector<double> > &A) {
    int lead = 0;

    for (int row=0; row<A.size(); ++row) {
        if (lead>A[row].size()) return;

        int i = row;
        while (A[i][lead]==0) {
            ++i;
            if (i>A.size()-1) {
                i = row;
                ++lead;
                if (lead>A[i].size()-1) return;
            }
        }

        // Swap the current row with the row having the non-zero leading value
        std::swap(A[i], A[row]);

        // Normalize the leading value to 1
        double div = A[row][lead];
        for (int j=0; j<A[row].size(); ++j) A[row][j] /= div;
    
        // Eliminate other entries in the leading column
        for (i=0; i<A.size(); ++i) {
            if (i!=row) {
                double mult = -A[i][lead];
                for (int j=0; j<A[i].size(); j++) A[i][j] += mult * A[row][j];
            }
        }
    }
}


void get_matrix(FILE *input, std::vector<std::vector<double> > &m) {

    char c;
    int i;
    double d;
    
    while (fscanf(input,"%lf",&d)==1) {

        m.push_back(std::vector<double>());
        m.back().push_back(d);
        
        while (fscanf(input,"%c",&c)==1) {
    
            if (c=='\n' || c=='\r') break;
            
            fscanf(input,"%lf",&d);
            m.back().push_back(d);
            
        }
        
    }
    
    // // sanity check
    
    // int length = m.size();
    
    // for (int i=0;i<m.size();i++) {
    
    //     assert(length==m[i].size() && "rref.cpp - get_matrix: read in mutation matrix that is not square");
        
    // }
    
}