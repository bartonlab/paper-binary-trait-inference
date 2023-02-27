#ifndef INF_H
#define INF_H

#include <vector>
#include <string>
#include <stdio.h>


// Typedefs
typedef std::vector<std::vector<double> > Vector;
typedef std::vector<std::vector<int> > IntVector;
typedef std::vector<std::vector<std::vector<double> > > VVector;
typedef std::vector<std::vector<std::vector<int> > > IntVVector;


// PROGRAM SETTINGS - This class holds the parameters needed for running the algorithm

class RunParameters {
    
public:
    
    std::string directory;             // path to the directory where the inut file is located
                                       // output will also be sent to this directory
    std::vector<std::string> infiles;  // input file list
    std::string muInfile;              // input file for mutation matrix
    std::string poInfile;              // input file for polygenic site information
    std::string poSequence;            // input file for polygenic sequence information
    std::string outfile;               // output file
    std::string covOutfile;            // output file for the regularized integrated covariance matrix
    std::string numOutfile;            // output file for the "numerator" (change in mutant frequency + mutation term)
    
    double tol;             // maximum tolerance for covariance differences before interpolating
    double gamma;           // Gaussian regularization strength
    double N;               // population size
    double mu;              // mutation rate per generation
    int q;                  // number of states for each allele
    
    bool useMatrix;         // if true, read mutation matrix from file
    bool useCovariance;     // if true, include covariance (linkage) information, else default to independent sites
    bool useAsymptotic;     // if true, assume that sequences are collected over long times (equilibrium)
    bool useVerbose;        // if true, print extra information while program is running
    bool saveCovariance;    // if true, output the total covariance matrix
    bool saveNumerator;     // if true, output the "numerator" multiplying the inverse covariance

    
    RunParameters() {
        
        directory = ".";
        muInfile  = "mu.dat";
        outfile   = "output.dat";
        poInfile  = "po.dat";
        poSequence= "poS.dat";
        
        tol   = 0.05;
        gamma = 1.0e3;
        N     = 1.0e3;
        mu    = 2.0e-4;
        q     = 2;
        
        useMatrix      = false;
        useCovariance  = true;
        useAsymptotic  = false;
        useVerbose     = false;
        saveCovariance = false;
        saveNumerator  = false;
        
    }
    std::string getSequenceInfile()              { return (directory+"/"+infiles[0]);  }
    std::string getSequenceInfile(int i)         { return (directory+"/"+infiles[i]);  }
    std::string getMuInfile()                    { return (directory+"/"+muInfile);    }
    std::string getPoInfile()                    { return (directory+"/"+poInfile);    }
    std::string getPoSInfile()                   { return (directory+"/"+poSequence);  }
    std::string getSelectionCoefficientOutfile() { return (directory+"/"+outfile);     }
    std::string getCovarianceOutfile()           { return (directory+"/"+covOutfile);  }
    std::string getNumeratorOutfile()            { return (directory+"/"+numOutfile);  }
    ~RunParameters() {}
    
};


// Main program
int run(RunParameters &r);

// Auxiliary routines
void computeAlleleFrequencies(const IntVector &sequences, const std::vector<double> &counts, const IntVector &poly_sites,const IntVector &poly_sequence,int q, std::vector<double> &p1, std::vector<double> &p2,std::vector<double> &pp);
void updateCovarianceIntegrate(double dg, const std::vector<double> &p1_0,const std::vector<double> &p2_0, const std::vector<double> &p1_1, const std::vector<double> &p2_1,double totalCov[]); 
void updateMuIntegrate(double dg, int L, const Vector &muMatrix, const IntVector &poly_sites,const IntVector &poly_sequence,const std::vector<double> &p1_0, const std::vector<double> &pp_0, const std::vector<double> &p1_1, const std::vector<double> &pp_1, std::vector<double> &totalMu);
void processStandard(const IntVVector &sequences, const Vector &counts, const std::vector<double> &times, const Vector &muMatrix, const IntVector &poly_sites,const IntVector &poly_sequence,int q, double totalCov[], double dx[]);
void regularizeCovariance(const IntVVector &sequences, int nP, int q, double gammaN, double totalCov[]);


#endif
