#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <chrono>

#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include "inf.h"    // inference declarations
#include "io.h"     // input/output

// typedef std::chrono::high_resolution_clock Clock;
bool useDebug = false;

// Compute single and pair allele frequencies from binary sequences and counts
void computeAlleleFrequencies(const IntVector &sequences,        // vector of sequence vectors in one generation
                              const std::vector<double> &counts, // vector of sequence counts
                              const IntVector &trait_sites,       // vector about escape site information
                              const IntVector &trait_sequence,    // vector about escape sequence information
                              int q,                   // number of states (e.g., number of nucleotides or amino acids)
                              std::vector<double> &p1, // single allele frequencies
                              std::vector<double> &p2, // pair allele frequencies
                              std::vector<double> &pt  // escape term frequencies
                              ) {

    // Set frequencies to zero
    for (int a=0;a<p1.size();a++) p1[a] = 0;
    for (int a=0;a<p2.size();a++) p2[a] = 0;
    for (int a=0;a<pt.size();a++) pt[a] = 0;

    int L  = (int) sequences[0].size(); //length of the genotype
    int LL = (int) p1.size(); // length of allele frequencies vector
    int ne = (int) trait_sites.size(); //number of escape group

    // Iterate through sequences and count the frequency of each state at each site,
    // and the frequency of each pair of states at each pair of sites

    for (int k=0;k<sequences.size();k++) { //genotypes in one generation, k:type

        std::vector<int> traitvalue(ne,0);

        for (int nn=0; nn<ne;nn++){

            for (int po=0;po<trait_sites[nn].size();po++){

                int po_site = (int)trait_sites[nn][po];
                traitvalue[nn] += abs(sequences[k][po_site] - trait_sequence[nn][po]);

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

            for (int nn=0; nn<ne;nn++){ if (traitvalue[nn] != 0) {

                    int bb = L * q + nn; // sequence length * allele number

                    p2[(a * LL) + bb] += counts[k];
                    p2[(bb * LL) + a] += counts[k];

            }}
        }

        for (int nn=0; nn<ne;nn++){ if (traitvalue[nn] != 0){

            int aa =  L * q + nn;//site for escape term

            p1[aa] += counts[k]; //single allele frequencies - escape part

            for (int mm=nn+1; mm<ne;mm++){ if (traitvalue[mm] != 0){

                int bb =  L * q + mm;

                p2[(aa * LL) + bb] += counts[k];
                p2[(bb * LL) + aa] += counts[k];

            }}

            int site = 0;
            int n_mutations = 0;//mutation number in the escape group

            for (int po=0; po<(int) trait_sites[nn].size(); po++){

                int po_site = trait_sites[nn][po];

                if (sequences[k][po_site] != trait_sequence[nn][po]){

                    n_mutations += 1;
                    site = po_site;

                }

            }
            if (n_mutations == 1){

                int qq = (int)sequences[k][site];
                pt[ ((nn * L) + site ) * q + qq] += counts[k];

            }
        }}
    }
}

// Calculate frequencies for recombination part (binary case)
void computeRecFrequencies(const IntVector &sequences,        // vector of sequence vectors in one generation
                           const std::vector<double> &counts, // vector of sequence counts
                           const IntVector &trait_sites,      // vector about escape site information
                           const IntVector &trait_sequence,   // vector about escape sequence information
                           std::vector<double>& pk            // frequencies for recombination part
                           ) {
    
    // Set frequencies to zero
    for (int a=0;a<pk.size();a++) pk[a] = 0;

    int L  = (int) sequences[0].size(); //length of the genotype, sequence length
    int LL = (int) pk.size(); // length of pk
    int ne = (int) trait_sites.size(); //number of escape group
    
    // Iterate through sequences and count the frequency for recombination term
    for (int n=0; n<ne;n++){ // for different trait groups
        
        int n_length = (int)trait_sites[n].size();             // length for escape group n
        Eigen::VectorXf sWT_n(n_length), sVec_n(n_length);
        
        // wild type sequence for escape group n 
        for (int i = 0; i < n_length; i++) sWT_n[i] = trait_sequence[n][i];    

        for (int k=0;k<sequences.size();k++) { //genotypes in one generation, k:type
            
            for (int ii = 0; ii < n_length; ii++) {
                int po_site = trait_sites[n][ii];                   // index for trait sites
                sVec_n[ii] = sequences[k][po_site]; // escape group n sequence for genotype_k
            }

            for (int nn = 0; nn < n_length - 1; nn++) {
                int k_bp = nn + 1;   // break point k
                int index_k = trait_sites[n][nn];                   // index for trait sites

                // MT before and after break point k
                if (sWT_n.head(k_bp) != sVec_n.head(k_bp) && sWT_n.tail(n_length - k_bp) != sVec_n.tail(n_length - k_bp)) {
                    pk[n * L * 3 + index_k * 3 + 0] += counts[k];
                }
                
                // # MT before break point k and WT after break point k
                if (sWT_n.head(k_bp) != sVec_n.head(k_bp) && sWT_n.tail(n_length - k_bp) == sVec_n.tail(n_length - k_bp)) {
                    pk[n * L * 3 + index_k * 3 + 1] += counts[k];
                }
                
                // # WT before break point k and MT after break point k
                if (sWT_n.head(k_bp) == sVec_n.head(k_bp) && sWT_n.tail(n_length - k_bp) != sVec_n.tail(n_length - k_bp)) {
                    pk[n * L * 3 + index_k * 3 + 2] += counts[k];
                }
            }
        }
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
                       const IntVector &trait_sites, // vector about escape information
                       const IntVector &trait_sequence,    // vector about escape sequence information
                       const std::vector<double> &p1_0, // single allele frequencies
                       const std::vector<double> &pt_0, // escape term frequencies
                       const std::vector<double> &p1_1, // single allele frequencies
                       const std::vector<double> &pt_1, // escape term frequencies
                       std::vector<double> &totalMu // contribution to selection estimate from mutation
                       ) {

    int  q = (int) muMatrix.size();   // number of tracked alleles (states)
    int ne = (int) trait_sites.size(); // number of escape goups

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

            totalMu[(i * q) + a] += dg * (fluxIn - fluxOut);

        }
    }

    for (int nn=0; nn<ne; nn++){

        for (int po=0;po<trait_sites[nn].size();po++){

            int  TF_index = trait_sequence[nn][po];
            int  po_site = (int) trait_sites[nn][po];

            for (int b=0;b<TF_index;b++) {

                double x_in  = 1 - 0.5 * ( p1_0[(L * q) + nn] + p1_1[(L * q) + nn]);
                double x_out = (pt_0[((nn * L) + po_site) * q + b]+pt_1[((nn * L) + po_site) * q + b])*0.5;
                totalMu[(L * q) + nn] += dg * (x_in * muMatrix[TF_index][b] - x_out * muMatrix[b][TF_index]);

            }

            for (int b=TF_index+1;b<q;b++) {

                double x_in  = 1 - 0.5 * ( p1_0[(L * q) + nn] + p1_1[(L * q) + nn]);
                double x_out = (pt_0[((nn * L) + po_site) * q + b]+pt_1[((nn * L) + po_site) * q + b])*0.5;
                totalMu[(L * q) + nn] += dg * (x_in * muMatrix[TF_index][b] - x_out * muMatrix[b][TF_index]);

            }
        }
    }
} 

void updateComIntegrate(double dg,                       // time step
                        int L,                           // sequence length
                        int q,                           // number of states (e.g., number of nucleotides or amino acids)
                        double r_rate,                   // recombination rate
                        const IntVector &trait_sites,    // vector about escape information
                        const IntVector &trait_sequence, // vector about escape sequence information
                        const IntVector &trait_dis,      // vector about escape sequence information
                        const std::vector<double> &p1_0, // single allele frequencies
                        const std::vector<double> &pk_0, // frequencies for recombination part
                        const std::vector<double> &p1_1, // single allele frequencies
                        const std::vector<double> &pk_1, // frequencies for recombination part
                        std::vector<double> &totalCom    // contribution to selection estimate from mutation
                        ) {

    int ne = (int) trait_sites.size(); // number of escape goups

    for (int n = 0; n < ne; n++) {
        
        double fluxIn  = 0;
        double fluxOut = 0;

        for (int nn = 0; nn < trait_sites[n].size() - 1; nn++) {

            int aa   = n * L * 3 + trait_sites[n][nn] * 3;
            fluxIn  += trait_dis[n][nn] * (1 - (p1_0[L * q + n]+p1_1[L * q + n])/2) * (pk_0[aa + 0]+pk_1[aa + 0])/2;
            fluxOut += trait_dis[n][nn] * (pk_0[aa + 1] + pk_1[aa + 1]) * (pk_0[aa + 2] + pk_1[aa + 2])/4;

        }
        totalCom[(L * q) + n] += dg * r_rate * (fluxIn - fluxOut);
    }
}

// Process standard sequences (time series)
void processStandard(const IntVVector &sequences,      // vector of sequence vectors
                     const Vector &counts,             // vector of sequence counts
                     const std::vector<double> &times, // sequence sampling times
                     const Vector &muMatrix,           // matrix of mutation rates
                     const IntVector &trait_sites,     // matrix of escape sites
                     const IntVector &trait_sequence,  // vector about escape sequence information
                     const IntVector &trait_dis,       // vector about escape sequence information
                     int q,                            // number of states (e.g., number of nucleotides or amino acids)
                     double r_rate,                    // recombination rate
                     double totalCov[],                // integrated covariance matrix
                     double dx[]                       // selection estimate numerator
                     ) {

    int L  = ((int) sequences[0][0].size());        // sequence length (i.e. number of tracked alleles)
    int ne = ((int) trait_sites.size());            // trait groups
    int LL =  L * q + ne;                           // length of allele frequency vector
    std::vector<double> totalMu(LL,0);              // accumulated mutation term
    std::vector<double> totalCom(LL,0);             // accumulated recombination term
    std::vector<double> p1(LL,0);                   // current allele frequency vector
    std::vector<double> p2(LL*LL,0);                // current allele pair frequencies
    std::vector<double> pt(L*ne*q,0);               // current escape term frequencies
    std::vector<double> pk(L*ne*3,0);               // current frequencies for recombination part
    std::vector<double> lastp1(LL,0);               // previous allele frequency vector
    std::vector<double> lastp2(LL*LL,0);            // previous allele pair frequencies
    std::vector<double> lastpt(L*ne*q,0);           // previous escape term frequencies
    std::vector<double> lastpk(L*ne*3,0);           // previous frequencies for recombination part

    // set initial allele frequency and covariance then loop
    computeAlleleFrequencies(sequences[0], counts[0], trait_sites, trait_sequence, q, lastp1, lastp2,lastpt);
    computeRecFrequencies(sequences[0], counts[0], trait_sites, trait_sequence,lastpk) ;

    for (int a=0;a<LL;a++) dx[a] -= lastp1[a]; // dx -= x[t_0]

    for (int k=1;k<sequences.size();k++) {

        computeAlleleFrequencies(sequences[k], counts[k], trait_sites, trait_sequence, q, p1, p2, pt);
        computeRecFrequencies(sequences[k], counts[k], trait_sites, trait_sequence,pk);
        updateCovarianceIntegrate(times[k] - times[k-1], lastp1, lastp2, p1, p2, totalCov);
        updateMuIntegrate(times[k] - times[k-1], L, muMatrix, trait_sites, trait_sequence, lastp1, lastpt, p1, pt, totalMu);
        updateComIntegrate(times[k] - times[k-1], L, q, r_rate, trait_sites, trait_sequence, trait_dis, lastp1, lastpk, p1, pk, totalCom);

        if (k==sequences.size()-1) { for (int a=0;a<LL;a++) dx[a] += p1[a]; }// dx += x[t_K]

        else { lastp1 = p1; lastp2 = p2; lastpt = pt; lastpk = pk;}

    }

    // Gather dx and totalMu terms
    for (int a=0;a<LL;a++) dx[a] -= (totalMu[a] + totalCom[a]);

    std::cout << "\nNumerator for escape part is\n";
    for (int a=0;a<ne;a++) std::cout << dx[L * q + a] << " ";

    std::cout << "\nNumerator for escape part without recombination part is\n";
    for (int a=0;a<ne;a++) std::cout << dx[L * q + a] - totalCom[L * q + a]<< " ";

    std::cout << "\nNumerator for escape part without mutation part is\n";
    for (int a=0;a<ne;a++) std::cout << dx[L * q + a] - totalMu[L * q + a]<< " ";

    std::cout << "\n";

}


// Add Gaussian regularization for selection coefficients (modifies integrated covariance)
void regularizeCovariance(const IntVVector &sequences, // vector of sequence vectors
                          int ne, //number of poly sites
                          int q, // number of states (e.g., number of nucleotides or amino acids)
                          double gamma, // normalized regularization strength
                          double totalCov[] // integrated covariance matrix
                          ) {

    int L = ((int) sequences[0][0].size()) ;
    int LL = L * q + ne;

    for (int a=0  ;a<L*q   ;a++) totalCov[(a * LL) + a] += gamma;    // standard regularization (individual part)
    for (int b=L*q;b<L*q+ne;b++) totalCov[(b * LL) + b] += gamma/10; // standard regularization (escape part)
    // for (int b=L*q;b<L*q+ne;b++) totalCov[(b * LL) + b] += gamma; // standard regularization (escape part)
}

// MAIN PROGRAM
int run(RunParameters &r) {
    // READ IN SEQUENCES FROM DATA
    IntVVector sequences;       // set of integer sequences at each time point
    Vector counts;              // counts for each sequence at each time point
    std::vector<double> times;  // list of times of measurements

    if (FILE *datain = fopen(r.getSequenceInfile().c_str(),"r")) { getSequences(datain, sequences, counts, times); fclose(datain); }
    else { printf("Problem retrieving data from file %s! File may not exist or cannot be opened.\n",r.getSequenceInfile().c_str()); return EXIT_FAILURE; }

    // MUTATION MATRIX
    Vector muMatrix;    
    if (r.useMatrix) { // from input file to get mutation matrix

        if (FILE *muin = fopen(r.getMuInfile().c_str(),"r")) { getMu(muin, muMatrix); fclose(muin); }
        else { printf("Problem retrieving data from file %s! File may not exist or cannot be opened.\n",r.getMuInfile().c_str()); return EXIT_FAILURE; }

        r.q = (int) muMatrix.size();

    }
    else { // no input file, use mutation rate mu to get the mutation matrix

        muMatrix.resize(r.q, std::vector<double>(r.q, r.mu));
        for (int i=0;i<r.q;i++) muMatrix[i][i] = 0;

    }

    // TRAIT INFORMATION (trait sites, TF trait sequences, distance between 2 trait sites)
    IntVector trait_sites; // trait sites
    if (FILE *poin = fopen(r.getTraitInfile().c_str(),"r"))  { getTrait(poin, trait_sites); fclose(poin);}
    else { printf("Problem retrieving data from file %s! File may not exist or cannot be opened.\n",r.getTraitInfile().c_str()); return EXIT_FAILURE; }

    IntVector trait_sequence; // TF trait sequences
    if (FILE *poin = fopen(r.getTraitSInfile().c_str(),"r")) { getTrait(poin, trait_sequence); fclose(poin);}
    else { printf("Problem retrieving data from file %s! File may not exist or cannot be opened.\n",r.getTraitSInfile().c_str()); return EXIT_FAILURE; }

    IntVector trait_dis; // TF trait sequences
    if (FILE *poin = fopen(r.getTraitDInfile().c_str(),"r")) { getTrait(poin, trait_dis); fclose(poin);}
    else { printf("Problem retrieving data from file %s! File may not exist or cannot be opened.\n",r.getTraitDInfile().c_str()); return EXIT_FAILURE; }

    // PROCESS SEQUENCES
    int    L         = ((int) sequences[0][0].size()) ;         // sequence length (i.e. number of tracked alleles)
    int    ne        = ((int) trait_sites.size());              // escape groups
    int    LL        =  L * r.q + ne;                           // length of allele frequency vector
    double tol       = r.tol;                                   // tolerance for changes in covariance between time points
    double gamma     = r.gamma;                                 // regularization strength for individual locus
    double r_rate    = r.rr;                                    // recombination rate
    double *dx       = new double[LL];                          // difference between start and end allele frequencies
    double *totalCov = new double[LL*LL];                       // accumulated allele covariance matrix

    for (int a=0;a<   LL;a++) dx[a]       = 0;
    for (int a=0;a<LL*LL;a++) totalCov[a] = 0;

    // _ START TIMER
    // auto t_start = Clock::now();

    processStandard(sequences, counts, times, muMatrix, trait_sites, trait_sequence, trait_dis, r.q, r.rr, totalCov, dx);

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
        processStandard(sequences, counts, times, muMatrix, trait_sites, trait_sequence, trait_dis, r.q, r.rr, totalCov, dx);

    } }

    // REGULARIZE
    regularizeCovariance(sequences, ne, r.q, gamma, totalCov);

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
