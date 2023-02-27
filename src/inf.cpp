#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <chrono>

#include "inf.h"    // inference declarations
#include "io.h"     // input/output


// typedef std::chrono::high_resolution_clock Clock;
bool useDebug = false;


// Compute single and pair allele frequencies from binary sequences and counts

void computeAlleleFrequencies(const IntVector &sequences,        // vector of sequence vectors in one generation
                              const std::vector<double> &counts, // vector of sequence counts
                              const IntVector &poly_sites,       // vector about escape site information
                              const IntVector &poly_sequence,    // vector about escape sequence information
                              int q,                   // number of states (e.g., number of nucleotides or amino acids)
                              std::vector<double> &p1, // single allele frequencies
                              std::vector<double> &p2, // pair allele frequencies
                              std::vector<double> &pp  // escape term frequencies
                              ) {

    // Set frequencies to zero

    for (int a=0;a<p1.size();a++) p1[a] = 0;
    for (int a=0;a<p2.size();a++) p2[a] = 0;
    for (int a=0;a<pp.size();a++) pp[a] = 0;

    int L  = (int) sequences[0].size(); //length of the genotype
    int LL = (int) p1.size(); // length of allele frequencies vector
    int nP = (int) poly_sites.size(); //number of escape group

    // Iterate through sequences and count the frequency of each state at each site,
    // and the frequency of each pair of states at each pair of sites

    for (int k=0;k<sequences.size();k++) { //genotypes in one generation, k:type

        std::vector<int> polyvalue(nP,0);

        for (int nn=0; nn<nP;nn++){

            for (int po=0;po<poly_sites[nn].size();po++){

                int po_site = (int)poly_sites[nn][po];
                polyvalue[nn] += abs(sequences[k][po_site] - poly_sequence[nn][po]);

            }

        }

        for (int i=0;i<sequences[k].size();i++) { // sites in one genotype, i:site

            int a = (i * q) + sequences[k][i]; // information include site i and allele in this site

            p1[a] += counts[k]; //single allele frequencies - selection part

            for (int j=i+1;j<sequences[k].size();j++) {

                int b = (j * q) + sequences[k][j]; // information include site i and allele in this site

                p2[(a * LL) + b] += counts[k]; // L not the length of the sequence but the length of p1 vector
                p2[(b * LL) + a] += counts[k]; //pair allele frequencies (symmetry matrix)
            }

            for (int nn=0; nn<nP;nn++){ if (polyvalue[nn] != 0) {

                    int bb = L * q + nn; // sequence length * allele number

                    p2[(a * LL) + bb] += counts[k];
                    p2[(bb * LL) + a] += counts[k];

            }}
        }

        for (int nn=0; nn<nP;nn++){ if (polyvalue[nn] != 0){

            int aa =  L * q + nn;//site for escape term

            p1[aa] += counts[k]; //single allele frequencies - escape part

            for (int mm=nn+1; mm<nP;mm++){ if (polyvalue[mm] != 0){

                int bb =  L * q + mm;

                p2[(aa * LL) + bb] += counts[k];
                p2[(bb * LL) + aa] += counts[k];

            }}

            int site = 0;
            int n_mutations = 0;//mutation number in the escape group

            for (int po=0; po<(int) poly_sites[nn].size(); po++){

                int po_site = poly_sites[nn][po];

                if (sequences[k][po_site] != poly_sequence[nn][po]){

                    n_mutations += 1;
                    site = po_site;

                }

            }
            if (n_mutations == 1){

                int qq = (int)sequences[k][site];
                pp[ ((nn * L) + site ) * q + qq] += counts[k];

            }

        }}
    }

}


// Update the summed covariance matrix

void updateCovarianceIntegrate(double dg, // time step
                               const std::vector<double> &p1_0, // single allele frequencies
                               const std::vector<double> &p2_0, // pair allele frequencies
                               const std::vector<double> &p1_1, // single allele frequencies
                               const std::vector<double> &p2_1, // pair allele frequencies
                               double totalCov[] // integrated covariance matrix
                               ) {

    int LL = (int) p1_0.size();

    // Iterate through states and add contributions to covariance matrix

    for (int a=0;a<LL;a++) {

        totalCov[(a * LL) + a] += dg * ( ((3 - (2 * p1_1[a])) * (p1_0[a] + p1_1[a])) - (2 * p1_0[a] * p1_0[a]) ) / 6;

        for (int b=a+1;b<LL;b++) {

            double dCov1 = -dg * ((2 * p1_0[a] * p1_0[b]) + (2 * p1_1[a] * p1_1[b]) + (p1_0[a] * p1_1[b]) + (p1_1[a] * p1_0[b])) / 6;
            double dCov2 = dg * 0.5 * (p2_0[(a * LL) + b] + p2_1[(a * LL) + b]);

            totalCov[(a * LL) + b] += dCov1 + dCov2;
            totalCov[(b * LL) + a] += dCov1 + dCov2;

        }

    }

}


// Update the summed mutation vector (flux out minus flux in)
// Note: since first row of mutation matrix is the reference, the mutation matrix is SHIFTED wrt frequencies,
// because the reference frequency is not explicitly tracked

void updateMuIntegrate(double dg, // time step
                       int L, //sequence length
                       const Vector &muMatrix, // mutation matrix
                       const IntVector &poly_sites, // vector about escape information
                       const IntVector &poly_sequence,    // vector about escape sequence information
                       const std::vector<double> &p1_0, // single allele frequencies
                       const std::vector<double> &pp_0, // single allele frequencies
                       const std::vector<double> &p1_1, // single allele frequencies
                       const std::vector<double> &pp_1, // single allele frequencies
                       std::vector<double> &totalMu // contribution to selection estimate from mutation
                       ) {

    int  q = (int) muMatrix.size();   // number of tracked alleles (states)
    int nP = (int) poly_sites.size(); // number of escape goups

    for (int i=0;i<L;i++) {

        for (int a=0;a<q;a++) {

            double fluxIn  = 0;
            double fluxOut = 0;

            for (int b=0;b<a;b++) {

                fluxIn  += 0.5 * (p1_0[(i * q) + b] + p1_1[(i * q) + b]) * muMatrix[b][a];
                fluxOut += 0.5 * (p1_0[(i * q) + a] + p1_1[(i * q) + a]) * muMatrix[a][b];

            }
            for (int b=a+1;b<q;b++) {

                fluxIn  += 0.5 * (p1_0[(i * q) + b] + p1_1[(i * q) + b]) * muMatrix[b][a];
                fluxOut += 0.5 * (p1_0[(i * q) + a] + p1_1[(i * q) + a]) * muMatrix[a][b];

            }

            totalMu[(i * q) + a] += dg * (fluxOut - fluxIn);

        }

    }

    for (int nn=0; nn<nP; nn++){

        for (int po=0;po<poly_sites[nn].size();po++){

            int  TF_index = poly_sequence[nn][po];
            int  po_site = (int) poly_sites[nn][po];


            for (int b=0;b<TF_index;b++) {

                double x_in  = 1 - 0.5 * ( p1_0[(L * q) + nn] + p1_1[(L * q) + nn]);
                double x_out = (pp_0[((nn * L) + po_site) * q + b]+pp_1[((nn * L) + po_site) * q + b])*0.5;
                totalMu[(L * q) + nn] += dg * (x_out * muMatrix[b][TF_index] - x_in * muMatrix[TF_index][b]);

            }

            for (int b=TF_index+1;b<q;b++) {

                double x_in  = 1 - 0.5 * ( p1_0[(L * q) + nn] + p1_1[(L * q) + nn]);
                double x_out = (pp_0[((nn * L) + po_site) * q + b]+pp_1[((nn * L) + po_site) * q + b])*0.5;
                totalMu[(L * q) + nn] += dg * (x_out * muMatrix[b][TF_index] - x_in * muMatrix[TF_index][b]);

            }

        }

    }

}


// Process standard sequences (time series)

void processStandard(const IntVVector &sequences, // vector of sequence vectors
                     const Vector &counts, // vector of sequence counts
                     const std::vector<double> &times, // sequence sampling times
                     const Vector &muMatrix, // matrix of mutation rates
                     const IntVector &poly_sites, // matrix of escape sites
                     const IntVector &poly_sequence,    // vector about escape sequence information
                     int q, // number of states (e.g., number of nucleotides or amino acids)
                     double totalCov[], // integrated covariance matrix
                     double dx[] // selection estimate numerator
                     ) {

    int L  = ((int) sequences[0][0].size());        // sequence length (i.e. number of tracked alleles)
    int nP = ((int) poly_sites.size());             // escape groups
    int LL =  L * q + nP;                           // length of allele frequency vector
    std::vector<double> totalMu(LL,0);              // accumulated mutation term
    std::vector<double> p1(LL,0);                   // current allele frequency vector
    std::vector<double> p2(LL*LL,0);                // current allele pair frequencies
    std::vector<double> pp(L*nP*q,0);                // current escape term frequencies
    std::vector<double> lastp1(LL,0);             // previous allele frequency vector
    std::vector<double> lastp2(LL*LL,0);          // previous allele pair frequencies
    std::vector<double> lastpp(L*nP*q,0);          // previous escape term frequencies

    // set initial allele frequency and covariance then loop
    computeAlleleFrequencies(sequences[0], counts[0], poly_sites, poly_sequence, q, lastp1, lastp2,lastpp);
    for (int a=0;a<LL;a++) dx[a] -= lastp1[a];

    for (int k=1;k<sequences.size();k++) {

        computeAlleleFrequencies(sequences[k], counts[k], poly_sites, poly_sequence, q, p1, p2, pp);
        updateCovarianceIntegrate(times[k] - times[k-1], lastp1, lastp2, p1, p2, totalCov);
        updateMuIntegrate(times[k] - times[k-1], L, muMatrix, poly_sites, poly_sequence, lastp1, lastpp, p1, pp, totalMu);

        if (k==sequences.size()-1) { for (int a=0;a<LL;a++) dx[a] += p1[a]; }//add last generation term
        else { lastp1 = p1; lastp2 = p2; lastpp = pp;}

    }

    // Gather dx and totalMu terms

    for (int a=0;a<LL;a++) dx[a] += totalMu[a];

}


// Add Gaussian regularization for selection coefficients (modifies integrated covariance)

void regularizeCovariance(const IntVVector &sequences, // vector of sequence vectors
                          int nP, //number of poly sites
                          int q, // number of states (e.g., number of nucleotides or amino acids)
                          double gammaN, // normalized regularization strength
                          double totalCov[] // integrated covariance matrix
                          ) {

    int L = ((int) sequences[0][0].size()) ;
    int LL = L * q + nP;

    for (int a=0  ;a<L*q   ;a++) totalCov[(a * LL) + a] += gammaN; // standard regularization (selection part)
    for (int b=L*q;b<L*q+nP;b++) totalCov[(b * LL) + b] += 1; // standard regularization (escape part)
    //for (int a=0;a<LL;a++) totalCov[(a * LL) + a] += gammaN; // standard regularization (no gauged state)

}


// MAIN PROGRAM

int run(RunParameters &r) {

    // READ IN SEQUENCES FROM DATA

    IntVVector sequences;       // set of integer sequences at each time point
    Vector counts;              // counts for each sequence at each time point
    std::vector<double> times;  // list of times of measurements

    if (FILE *datain = fopen(r.getSequenceInfile().c_str(),"r")) { getSequences(datain, sequences, counts, times); fclose(datain); }
    else { printf("Problem retrieving data from file %s! File may not exist or cannot be opened.\n",r.getSequenceInfile().c_str()); return EXIT_FAILURE; }

    Vector muMatrix;    // matrix of mutation rates

    if (r.useMatrix) {

        if (FILE *muin = fopen(r.getMuInfile().c_str(),"r")) { getMu(muin, muMatrix); fclose(muin); }
        else { printf("Problem retrieving data from file %s! File may not exist or cannot be opened.\n",r.getMuInfile().c_str()); return EXIT_FAILURE; }

        r.q = (int) muMatrix.size();

    }
    else {

        muMatrix.resize(r.q, std::vector<double>(r.q, r.mu));
        for (int i=0;i<r.q;i++) muMatrix[i][i] = 0;

    }

    IntVector poly_sites; //sites that are escape groups

    if (FILE *poin = fopen(r.getPoInfile().c_str(),"r")) { getPo(poin, poly_sites); fclose(poin);}
    else { printf("Problem retrieving data from file %s! File may not exist or cannot be opened.\n",r.getPoInfile().c_str()); return EXIT_FAILURE; }

    IntVector poly_sequence; //sites that are escape groups

    if (FILE *poin = fopen(r.getPoSInfile().c_str(),"r")) { getPoSequences(poin, poly_sequence); fclose(poin);}
    else { printf("Problem retrieving data from file %s! File may not exist or cannot be opened.\n",r.getPoSInfile().c_str()); return EXIT_FAILURE; }


    // PROCESS SEQUENCES

    int    L         = ((int) sequences[0][0].size()) ;         // sequence length (i.e. number of tracked alleles)
    int    nP        = ((int) poly_sites.size());               // escape groups
    int    LL        =  L * r.q + nP;                            // length of allele frequency vector
    double tol       = r.tol;                                   // tolerance for changes in covariance between time points
    double gammaN    = r.gamma/r.N;                             // regularization strength divided by population size
    double *dx       = new double[LL];                          // difference between start and end allele frequencies
    double *totalCov = new double[LL*LL];                       // accumulated allele covariance matrix

    for (int a=0;a<   LL;a++) dx[a]       = 0;
    for (int a=0;a<LL*LL;a++) totalCov[a] = 0;

    // _ START TIMER
    // auto t_start = Clock::now();

    //if (r.useAsymptotic) processAsymptotic(sequences, counts, muMatrix, r.q, totalCov, dx);
    //else
    processStandard(sequences, counts, times, muMatrix, poly_sites, poly_sequence, r.q, totalCov, dx);

    // If there is more than one input trajectory, loop through all of them and add contributions
    // NOTE: CURRENT CODE ASSUMES THAT ALL VALUES OF N ARE EQUAL

    if (r.infiles.size()>1) { for (int k=1;k<r.infiles.size();k++) {

        // Reset trajectory variables and reload them with new data

        sequences.clear();
        counts.clear();
        times.clear();

        if (FILE *datain = fopen(r.getSequenceInfile(k).c_str(),"r")) { getSequences(datain, sequences, counts, times); fclose(datain); }
        else { printf("Problem retrieving data from file %s! File may not exist or cannot be opened.\n",r.getSequenceInfile().c_str()); return EXIT_FAILURE; }

        // Add contributions to dx and totalCov

        processStandard(sequences, counts, times, muMatrix, poly_sites, poly_sequence, r.q, totalCov, dx);

    } }

    // REGULARIZE

    regularizeCovariance(sequences, nP, r.q, gammaN, totalCov);

    // RECORD COVARIANCE (optional)

    if (r.saveCovariance) {
        if (FILE *dataout = fopen(r.getCovarianceOutfile().c_str(),"w")) { printCovariance(dataout, totalCov, LL); fclose(dataout); }
        else { printf("Problem writing data to file %s! File may not be created or cannot be opened.\n",r.getCovarianceOutfile().c_str()); return EXIT_FAILURE; }
    }

    // RECORD NUMERATOR (optional)

    if (r.saveNumerator) {
        if (FILE *dataout = fopen(r.getNumeratorOutfile().c_str(),"w")) { printNumerator(dataout, dx, LL); fclose(dataout); }
        else { printf("Problem writing data to file %s! File may not be created or cannot be opened.\n",r.getCovarianceOutfile().c_str()); return EXIT_FAILURE; }
    }

    // INFER THE SELECTION COEFFICIENTS -- solve Cov . sMAP = dx

    std::vector<double> sMAP(LL,0);

    if (r.useCovariance) {

        int status;

        gsl_matrix_view _cov = gsl_matrix_view_array(totalCov, LL, LL);   // gsl covariance + Gaussian regularization
        gsl_vector_view  _dx = gsl_vector_view_array(dx, LL);            // gsl dx vector
        gsl_vector    *_sMAP = gsl_vector_alloc(LL);                     // maximum a posteriori selection coefficients for each allele
        gsl_permutation  *_p = gsl_permutation_alloc(LL);

        gsl_linalg_LU_decomp(&_cov.matrix, _p, &status);
        gsl_linalg_LU_solve(&_cov.matrix, _p, &_dx.vector, _sMAP);

        for (int a=0;a<LL;a++) sMAP[a] = gsl_vector_get(_sMAP, a);

        gsl_permutation_free(_p);
        gsl_vector_free(_sMAP);

        delete[] dx;
        delete[] totalCov;

    }

    else {

        for (int a=0;a<LL;a++) sMAP[a] = dx[a] / totalCov[(a * LL) + a];

    }

    // auto t_end = Clock::now();
    // ^ END TIMER

    // printf("%lld\n",std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count());

    // WRITE TO FILE

    if (FILE *dataout = fopen(r.getSelectionCoefficientOutfile().c_str(),"w")) { printSelectionCoefficients(dataout, sMAP); fclose(dataout); }
    else { printf("Problem writing data to file %s! File may not be created or cannot be opened.\n",r.getSelectionCoefficientOutfile().c_str()); return EXIT_FAILURE; }

    if (r.useVerbose) {
        int lwidth = 5;
        printf("s = {\t");
        for (int a=0;a<LL;a++) { if (a%lwidth==0 && a>0) printf("\n\t"); printf("%.4e\t",sMAP[a]); }
        printf("}\n");
    }

    return EXIT_SUCCESS;

}
