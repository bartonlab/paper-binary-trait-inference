g++ binary/main.cpp binary/inf_binary.cpp binary/io_binary.cpp -march=native -lgsl -lgslcblas -o mpl
./mpl -d ../data/simulation/example -i example-0_ns1000_dt1.dat -o sc-0_ns1000_dt1-C.dat -g 1 -N 1e3 -mu 2e-4 -rr 2e-4 -e traitsite-0.dat -es traitseq-0.dat -ed traitdis-0.dat -sc covariance-0_ns1000_dt1-C.dat
