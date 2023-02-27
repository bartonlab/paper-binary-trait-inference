import numpy as np
try:
    import itertools.izip as zip
except ImportError:
    import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import random

# GitHub
HIV_DIR = 'data/HIV'
SIM_DIR = 'data/simulation'
FIG_DIR = 'figures'


def simulate(**pdata):
    """
    Example evolutionary trajectory for a 50-site system
    """

    # unpack passed data

    n_gen    = pdata['n_gen']
    N        = pdata['N']
    po_site  = pdata['po_site']
    alphabet = pdata['alphabet']
    xfile    = pdata['xfile']
    output   = pdata['output']
    t_format = pdata['t_format']
    d_format = pdata['d_format']

    mu    = pdata['mu']
    n_ben = pdata['n_ben']
    n_neu = pdata['n_neu']
    n_del = pdata['n_del']
    n_pol = pdata['n_pol']
    s_ben = pdata['s_ben']
    s_neu = pdata['s_neu']
    s_del = pdata['s_del']
    s_pol = pdata['s_pol']


    q     = len(alphabet)
    seq_length = n_ben+n_neu+n_del
    muMatrix = np.ones([q,q]) * mu
    for i in range(q): muMatrix[i][i] = 0 # make sure the diagonal equals to 0

    # FUNCTIONS
    # use number to represent sequence (previous: alphabet)
    def get_genotype_value(genotype):
        x = np.zeros(seq_length)
        for i in range(seq_length):
            for n in range(q):
                if genotype[i] == alphabet[n]:
                    x[i] = n
        return x

    # get fitness of new genotype (number format)
    def get_fitness_number(genotype):
        fitness[genotype] = 1;
        geno_value = get_genotype_value(genotype)
        # selection part
        h = np.zeros(seq_length)
        for i in range(n_ben): h[i]      = s_ben;
        for i in range(n_del): h[-(i+1)] = s_del;
        fitness[genotype] += np.sum(h * geno_value);
        # trait part
        for n in range(n_pol):
            poly_n = 0;
            for j in po_site[n]:
                poly_n += abs(geno_value[j] - alphabet.in_delex('A'));
            if poly_n != 0:
                fitness[genotype] += s_pol;
        return fitness[genotype]

    # get fitness of new genotype (alphabet format)
    def get_fitness_alpha(genotype):
        fitness[genotype] = 1;
        #alphabet format
        for i in range(n_ben):
            if genotype[i] != "A":
                fitness[genotype] += s_ben
        for i in range(n_del):
            if genotype[-(i+1)] != "A":
                fitness[genotype] += s_del
        for n in range(n_pol):
            for j in po_site[n]:
                if genotype[j] != "A":
                    fitness[genotype] += s_pol
                    break
        return fitness[genotype]

    # for different genotypes, they have different mutation probablity
    # this total mutation value represents the likelihood for one genotype to mutate
    # if there is only 2 alleles an_del only one mutation rate, total_mu = mutation rate * sequence length
    def get_mu(genotype):
        total_mu = 0
        geno_value = get_genotype_value(genotype)
        for i in range(seq_length):
            a = int(geno_value[i])
            mu_i = sum(muMatrix[a]) # calculate total mutation rate in site i
            total_mu += mu_i
        return total_mu

    # take a supplied genotype an_del mutate a site at random.
    def get_mutant(genotype):
       #choose one possible mutation site
       site = np.random.randint(seq_length);
       #choose possible mutation base
       geno_value = get_genotype_value(genotype);
       a = int(geno_value[site]);
       mu_fre = muMatrix[a];
       mu_tot = sum(mu_fre);
       mu_f = [x / mu_tot for x in mu_fre];
       # choose the mutation allele according to mutation rate (for multiple allele case)
       mutation = np.random.choice(alphabet, p=mu_f);
       # get new mutation sequence
       new_genotype = genotype[:site] + mutation[0] + genotype[site+1:];
       return new_genotype

    # check if the mutant already exists in the population.
    #If it does, increment this mutant genotype;
    #If it doesn't create a new genotype of count 1.
    # If a mutation event creates a new genotype, calculate its fitness.
    def mutation_event(genotype,pop,fitness):
        if pop[genotype] > 1:
            pop[genotype] -= 1
            if pop[genotype] == 0:
                del pop[genotype]

            new_genotype = get_mutant(genotype)
            if new_genotype in pop:
                pop[new_genotype] += 1
            else:
                pop[new_genotype] = 1
            if new_genotype not in fitness:
                fitness[new_genotype] = get_fitness_alpha(new_genotype)
        return pop,fitness

    def mutation_step(pop,fitness):
        genotypes = list(pop.keys())
        for genotype in genotypes:
            n = pop[genotype]
            # calculate the likelihood to mutate
            total_mu =  mu * seq_length # for binary case
            # total_mu = get_mu(genotype) # for multiple case (mutation rate is different for alleles)
            nMut = np.random.binomial(n, total_mu)
            for j in range(nMut):
                mutation_event(genotype,pop,fitness);
        return pop,fitness

    def offspring_step(pop,fitness):
        genotypes = list(pop.keys())
        numbers = [x for x in pop.values()]
        fitnesses = [fitness[genotype] for genotype in genotypes]
        weights = [x * y for x,y in zip(numbers, fitnesses)]
        total = sum(weights)
        weights = [x / total for x in weights]
        counts = list(np.random.multinomial(N, weights)) # genetic drift
        for (genotype, count) in zip(genotypes, counts):
            if (count > 0):
                pop[genotype] = count
            else:
                del pop[genotype]
        return pop,fitness

    # Simulation

    # initial genotype
    pop = {}
    pop["AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"] = N
    fitness = {}
    fitness["AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"] = 1.0
    history = [];
    clone_pop = dict(pop)
    history.append(clone_pop)

    ## in every generation, it will mutate an_del then the genetic drift
    ## calculate several times to get the evolution trajectory
    ## At each step in the simulation, we append to a history object.
    for i in range(n_gen):
        # create all the mutations that occur in one single generation
        mutation_step(pop,fitness)
        # get genetic drift occurring in one generation
        offspring_step(pop,fitness)
        clone_pop = dict(pop)
        history.append(clone_pop)

    # Write output file
    ## txt format ---- python code
    if t_format == True:
        file = open("./%s/%s.txt"%(SIM_DIR,xfile),'w');
        file.writelines(str(history));
        file.close();

    ## dat format ---- C++ code
    if d_format == True:
        f = open("./%s/%s.dat"%(SIM_DIR,xfile),'w')
        for i in range(len(history)):
            pop_at_t = history[i]
            genotypes = pop_at_t.keys()
            for genotype in genotypes:
                times = i
                counts = pop_at_t[genotype]
                sequence = []
                sequence.append([str(alphabet.index(a)) for a in genotype])
                f.write('%d\t%d\t%s\n' % (times,counts, ' '.join(sequence[0])))
        f.close()

    if output == True:
        print('Simulation completes')
        print('beginning with wild type, sequence has %d types of alleles, totally run %d generations' %(q,n_gen))
        print('the length of the sequence is %d, with %d beneficial sites (%s) and %d deleterious sites（%s)'%(seq_length,n_ben,s_ben,n_del,s_del))
        print('containing %d trait groups with trait coefficients equal to %s, they are '%(n_pol,s_pol))
        print(po_site)

def run_mpl_multiple(**pdata):

    """
    Use mpl to calculate selection coefficients and trait coefficients (binary case)
    """

    # unpack passed data

    n_gen    = pdata['n_gen']
    N        = pdata['N']
    po_site  = pdata['po_site']
    alphabet = pdata['alphabet']
    xfile    = pdata['xfile']
    yfile    = pdata['yfile']
    output   = pdata['output']

    mu    = pdata['mu']
    n_ben = pdata['n_ben']
    n_neu = pdata['n_neu']
    n_del = pdata['n_del']
    n_pol = pdata['n_pol']
    s_ben = pdata['s_ben']
    s_neu = pdata['s_neu']
    s_del = pdata['s_del']
    s_pol = pdata['s_pol']
    gamma = pdata['gamma']

    q     = len(alphabet)
    seq_length  = n_ben+n_neu+n_del
    x_length = seq_length*q + n_pol  #binary case

    # FUNCTIONS
    # use number to represent sequence (previous: alphabet)
    def get_genotype_value(genotype):
        x = np.zeros(seq_length)
        for i in range(seq_length):
            for n in range(q):
                if genotype[i] == alphabet[n]:
                    x[i] = n
        return x

    # determine if one epitope is mutant
    def get_poly_value(genotype):
        p = np.zeros(n_pol)
        for n in range(n_pol):
            for j in po_site[n]:
                if genotype[j] != "A":
                    p[n] = 1
        return p

    # get allele frequency x_i
    def get_mutant_frequency(pop_at_t):
        genotypes = pop_at_t.keys()
        x = np.zeros(x_length)
        for genotype in genotypes:
            genovalue = get_genotype_value(genotype)
            polyvalue = get_poly_value(genotype)
            for i in range(seq_length):
                aa = i * q +  int(genovalue[i])
                x[aa] += pop_at_t[genotype]
            for n in range(n_pol):
                aa =  seq_length * q + n
                if polyvalue[n] == 1:
                    x[aa] += pop_at_t[genotype]
        x_frequencies = x / float(N)
        return x_frequencies

    # calculate single an_del pair allele frequency (binary case)
    def get_allele_frequency(pop_at_t):
        genotypes = pop_at_t.keys()
        x  = np.zeros(x_length)             # single allele frequency
        xx = np.zeros([x_length,x_length]) # pair allele frequency
        for genotype in genotypes:
            genovalue = get_genotype_value(genotype)
            polyvalue = get_poly_value(genotype)
            # selection part
            for i in range(seq_length):
                aa = i * q +  int(genovalue[i])
                x[aa] += pop_at_t[genotype]
                for j in range(int(i+1), seq_length):
                    bb = j * q +  int(genovalue[j])
                    xx[aa,bb] += pop_at_t[genotype]
                    xx[bb,aa] += pop_at_t[genotype]
                for n in range(n_pol):
                    if polyvalue[n] == 1:
                        bb = seq_length * q + n
                        xx[aa,bb] += pop_at_t[genotype]
                        xx[bb,aa] += pop_at_t[genotype]
            # trait part
            for n in range(n_pol):
                aa = seq_length * q + n
                if polyvalue[n] == 1:
                    x[aa] += pop_at_t[genotype]
                    for m in range(n_pol):
                        bb = seq_length * q + m
                        if polyvalue[m] == 1:
                            xx[aa,bb] += pop_at_t[genotype]
        x_frequencies = x / float(N)
        xx_frequencies = xx / float(N)
        return x_frequencies,xx_frequencies

    # diffusion matrix C
    def diffusion_matrix_at_t(pop_at_t):
        C = np.zeros([x_length,x_length])
        x_frequencies, xx_frequencies = get_allele_frequency(pop_at_t)
        for i in range(x_length):
            C[i,i] = x_frequencies[i] - x_frequencies[i] * x_frequencies[i]
            for j in range(i+1 ,x_length):
                C[i,j] = xx_frequencies[i,j] - x_frequencies[i] * x_frequencies[j]
                C[j,i] = xx_frequencies[i,j] - x_frequencies[i] * x_frequencies[j]
        return C

    # get denominator
    def get_first_term(history):
        C_int = np.zeros([x_length,x_length])
        for k in range(n_gen-1):
            pop_at_t = history[k] #modify
            C_int +=  diffusion_matrix_at_t(pop_at_t)
        I = np.identity(x_length)
        denominator = np.mat(C_int + I * gamma)
        return np.linalg.inv(denominator)# get the inverse matrix

    # get the new term xx for trait term
    def get_mu_frequency(pop_at_t,po_site):
        genotypes = pop_at_t.keys()
        x_mu = np.zeros([seq_length,q])
        for genotype in genotypes:
            genovalue = get_genotype_value(genotype)
            n_mutations = 0
            for j in po_site:
                if genotype[j] != "A":
                    n_mutations += 1
                    site = j
            if n_mutations == 1:
                qq = int(genovalue[site])
                x_mu[site,qq] += pop_at_t[genotype]
        x_mu = x_mu/ float(N)
        return x_mu


    def get_flux(pop_at_t):
        x1 = get_mutant_frequency(pop_at_t)
        flux = np.zeros(x_length)
        for i in range(seq_length):
            for a in range(q):
                for b in range(q):
                    if b != a:
                        flux[i*q+a] += mu*(x1[i*q+b]-x1[i*q+a]);
        for n in range(n_pol):
            x2 = get_mu_frequency(pop_at_t,po_site[n])
            for j in po_site[n]:
                for b in range(1,q):
                    x_in = 1 - x1[seq_length * q + n]
                    x_out = x2[j,b]
                    flux[seq_length*q+n] +=  mu * (x_in - x_out)
        return flux

    def get_second_term(history):
        numerator = np.zeros([x_length,1])
        flux = np.zeros(x_length)
        pop_at_K = history[n_gen]
        pop_at_0 = history[0]
        x_tK = get_mutant_frequency(pop_at_K)
        x_t0 = get_mutant_frequency(pop_at_0)
        for k in range(n_gen-1):
            pop_at_t = history[k]
            flux += get_flux(pop_at_t)
        nu = x_tK - x_t0 - flux
        for i in range(len(nu)):
            numerator[i,0] = nu[i]
        return numerator

    #run mpl
    history = {}
    file = open("./%s/%s.txt"%(SIM_DIR,xfile),'r+');#modify
    history = eval(file.read())
    file.close();
    c_all = np.mat(get_first_term(history))*np.mat(get_second_term(history))
    sc_all = np.zeros(seq_length*(q-1)+n_pol)
    for j in range(seq_length):
        for a in range(q-1):
            index = j * (q - 1) + a
            sc_all[index] = c_all[j*q+a+1] - c_all[j*q]
    for n in range(n_pol):
        index = seq_length*(q - 1) + n
        sc_all[index] = c_all[seq_length*q+n]

    np.savetxt("./%s/sc-%s.dat"%(SIM_DIR,yfile),sc_all)

    if output == True:
        print('Calculation completes')
        print('inferred beneficial selection coefficients are:')
        print(sc_all[0:n_ben])
        print('inferred neutral selection coefficients are:')
        print(sc_all[n_ben:n_ben+n_neu])
        print('inferred deleterious selection coefficients are:')
        print(sc_all[n_ben+n_neu:seq_length])
        print('inferred trait coefficients are:')
        print(c_all[seq_length*q:])

def run_mpl_binary(**pdata):

    """
    Use mpl to calculate selection coefficients and trait coefficients (binary case)
    """

    # unpack passed data

    n_gen    = pdata['n_gen']
    N        = pdata['N']
    po_site  = pdata['po_site']
    alphabet = pdata['alphabet']
    xfile    = pdata['xfile']
    yfile    = pdata['yfile']
    save_cov = pdata['save_cov']
    output = pdata['output']

    mu    = pdata['mu']
    n_ben = pdata['n_ben']
    n_neu = pdata['n_neu']
    n_del = pdata['n_del']
    n_pol = pdata['n_pol']
    s_ben = pdata['s_ben']
    s_neu = pdata['s_neu']
    s_del = pdata['s_del']
    s_pol = pdata['s_pol']
    gamma = pdata['gamma']

    q     = len(alphabet)
    seq_length  = n_ben+n_neu+n_del
    x_length = seq_length + n_pol  #binary case

    # FUNCTIONS
    # use number to represent sequence (previous: alphabet)
    def get_genotype_value(genotype):
        x = np.zeros(seq_length)
        for i in range(seq_length):
            for n in range(q):
                if genotype[i] == alphabet[n]:
                    x[i] = n
        return x

    # determine if one epitope is mutant
    def get_poly_value(genotype):
        p = np.zeros(n_pol)
        for n in range(n_pol):
            for j in po_site[n]:
                if genotype[j] != "A":
                    p[n] = 1
        return p

    # get allele frequency x_i
    def get_mutant_frequency(pop_at_t):
        genotypes = pop_at_t.keys()
        x = np.zeros(x_length)
        for genotype in genotypes:
            genovalue = get_genotype_value(genotype)
            polyvalue = get_poly_value(genotype)
            for i in range(seq_length):
                if genovalue[i] != 0:
                    x[i] += pop_at_t[genotype]
            for n in range(n_pol):
                if polyvalue[n] != 0:
                    x[seq_length + n] += pop_at_t[genotype]
        x_frequencies = x / float(N)
        return x_frequencies

    # calculate single an_del pair allele frequency (binary case)
    def get_allele_frequency(pop_at_t):
        genotypes = pop_at_t.keys()
        x = np.zeros(x_length)             # single allele frequency
        xx = np.zeros([x_length,x_length]) # pair allele frequency
        for genotype in genotypes:
            genovalue = get_genotype_value(genotype)
            polyvalue = get_poly_value(genotype)
            # selection part
            for i in range(seq_length):
                if genovalue[i] != 0:
                    x[i] += pop_at_t[genotype]
                    for j in range(int(i+1), seq_length):
                        if genovalue[j] != 0:
                            xx[i,j] += pop_at_t[genotype]
                            xx[j,i] += pop_at_t[genotype]
                    for n in range(n_pol):
                        if polyvalue[n] == 1:
                            xx[i,seq_length + n] += pop_at_t[genotype]
                            xx[seq_length + n,i] += pop_at_t[genotype]
            # trait part
            for n in range(n_pol):
                aa = seq_length + n
                if polyvalue[n] == 1:
                    x[aa] += pop_at_t[genotype]
                    for m in range(n_pol):
                        bb = seq_length + m
                        if polyvalue[m] == 1:
                            xx[aa,bb] += pop_at_t[genotype]
        x_frequencies = x / float(N)
        xx_frequencies = xx / float(N)
        return x_frequencies,xx_frequencies

    # diffusion matrix C
    def diffusion_matrix_at_t(pop_at_t):
        C = np.zeros([x_length,x_length])
        x_frequencies, xx_frequencies = get_allele_frequency(pop_at_t)
        for i in range(x_length):
            C[i,i] = x_frequencies[i] - x_frequencies[i] * x_frequencies[i]
            for j in range(i+1 ,x_length):
                C[i,j] = xx_frequencies[i,j] - x_frequencies[i] * x_frequencies[j]
                C[j,i] = xx_frequencies[i,j] - x_frequencies[i] * x_frequencies[j]
        return C

    # get denominator
    def get_first_term(history):
        C_int = np.zeros([x_length,x_length])
        for k in range(n_gen-1):
            pop_at_t = history[k] #modify
            C_int +=  diffusion_matrix_at_t(pop_at_t)
        I = np.identity(x_length)
        denominator = np.mat(C_int + I * gamma)
        if save_cov == True:
            np.savetxt("./%s/example_covariance.dat"%SIM_DIR,C_int)
        return np.linalg.inv(denominator)# get the inverse matrix

    # get the new term xx for trait term
    def get_mu_frequency(pop_at_t,po_site):
        genotypes = pop_at_t.keys()
        x_mu = np.zeros(seq_length)
        for genotype in genotypes:
            genovalue = get_genotype_value(genotype)
            n_mutations = 0
            for j in po_site:
                if genotype[j] != "A":
                    n_mutations += 1
                    site = j
            if n_mutations == 1:
                qq = int(genovalue[site])
                x_mu[site] += pop_at_t[genotype]
        x_mu = x_mu/ float(N)
        return x_mu


    def get_flux(pop_at_t):
        x1 = get_mutant_frequency(pop_at_t)
        flux = np.zeros(x_length)
        for i in range(seq_length):
            flux[i] = mu * ( 1 - 2 * x1[i])
        for n in range(n_pol):
            x2 = get_mu_frequency(pop_at_t,po_site[n])
            for j in po_site[n]:
                x_in = 1 - x1[seq_length + n]
                x_out = x2[j]
                flux[seq_length+n] +=  mu * (x_in - x_out)
        return flux

    def get_second_term(history):
        numerator = np.zeros([x_length,1])
        flux = np.zeros(x_length)
        pop_at_K = history[n_gen]
        pop_at_0 = history[0]
        x_tK = get_mutant_frequency(pop_at_K)
        x_t0 = get_mutant_frequency(pop_at_0)
        for k in range(n_gen-1):
            pop_at_t = history[k]
            flux += get_flux(pop_at_t)
        nu = x_tK - x_t0 - flux
        for i in range(len(nu)):
            numerator[i,0] = nu[i]
        return numerator

    #run mpl
    history = {}
    file = open("./%s/%s.txt"%(SIM_DIR,xfile),'r+');#modify
    history = eval(file.read())
    file.close();
    c_all = np.mat(get_first_term(history))*np.mat(get_second_term(history))
    np.savetxt("./%s/sc_%s.dat"%(SIM_DIR,yfile),c_all)

    if output == True:
        print('Calculation completes')
        print('inferred beneficial selection coefficients are:')
        print(c_all[0:n_ben])
        print('inferred neutral selection coefficients are:')
        print(c_all[n_ben:n_ben+n_neu])
        print('inferred deleterious selection coefficients are:')
        print(c_all[n_ben+n_neu:seq_length])
        print('inferred trait coefficients are:')
        print(c_all[seq_length:])

def py2c(**pdata):

    """
    Convert a trajectory into plain text to save the results.
    """

    # unpack passed data

    t0       = pdata['t0']
    T        = pdata['T']
    ns_vals  = pdata['ns']
    dt_vals  = pdata['dt']
    xfile    = pdata['xfile']
    alphabet = pdata['alphabet']

    rng = np.random.RandomState()

    # FUNCTIONS --  process input files
    def get_sequence(t,input):
        seq_s = []
        seq_n = []
        for i in range(len(input)):
            if input[i][0] == t:
                s_t = ''
                seq_n.append(int(input[i][1]))
                for j in range(len(input[i])-2):
                    s_t += str(int(input[i][j+2]))
                seq_s.append(str(s_t))
        return seq_s, seq_n

    def choose_input(pop,n_s):
        genotypes = list(pop.keys())
        numbers = [x for x in pop.values()]
        new_genotypes = random.choices(genotypes, weights = numbers, k = n_s)
        new_pop = {};
        for genotype in new_genotypes:
            if genotype in new_pop:
                new_pop[genotype] += 1;
            else:
                new_pop[genotype] = 1;
        return new_pop

    # write the results
    for i in range(len(ns_vals)):
        ns      = ns_vals[i]
        for j in range(len(dt_vals)):
            dt = dt_vals[j]
            if dt == 1 and ns != 1000:
                f = open('%s/jobs/seq/%s_ns%d_dt%d.dat' % (SIM_DIR, xfile, ns, dt), 'w')
                history = {}
                file = open('%s/jobs/seq/%s_ns1000_dt1.txt'%(SIM_DIR, xfile));
                history = eval(file.read())
                for t in range(T+1):
                    pop_at_t = history[t]
                    pop_t = choose_input(pop_at_t,ns)
                    genotypes = pop_t.keys()
                    for genotype in genotypes:
                        counts = pop_t[genotype]
                        f.write('%d\t%d\t' % (t,counts))
                        f.write('%s\n' % (' '.join([str(alphabet.index(genotype[j])) for j in range(len(genotype))])))
            elif ns == 1000 and dt == 1:
                continue
            else:
                f = open('%s/jobs/seq/%s_ns%d_dt%d.dat' % (SIM_DIR, xfile, ns, dt), 'w')
                input = np.loadtxt('%s/jobs/seq/%s_ns%d_dt1.dat'%(SIM_DIR, xfile,ns))
                for t in range(t0,T+1,dt):
                    seq_s,seq_n = get_sequence(t,input)
                    for k in range(len(seq_n)):
                        f.write('%d\t%d\t%s\n' %(t, seq_n[k], ' '.join([str(kk) for kk in seq_s[k]])))
            f.close()

def run_mpl_binary_old(**pdata):

    """
    Use mpl to calculate selection coefficients and trait coefficients (binary case)
    """

    # unpack passed data

    n_gen    = pdata['n_gen']
    N        = pdata['N']
    po_site  = pdata['po_site']
    alphabet = pdata['alphabet']
    xfile    = pdata['xfile']
    yfile    = pdata['yfile']
    output = pdata['output']

    mu    = pdata['mu']
    n_ben = pdata['n_ben']
    n_neu = pdata['n_neu']
    n_del = pdata['n_del']
    s_ben = pdata['s_ben']
    s_neu = pdata['s_neu']
    s_del = pdata['s_del']
    gamma = pdata['gamma']

    q     = len(alphabet)
    seq_length  = n_ben+n_neu+n_del
    x_length = seq_length  #binary case without trait term

    # FUNCTIONS
    # use number to represent sequence (previous: alphabet)
    def get_genotype_value(genotype):
        x = np.zeros(seq_length)
        for i in range(seq_length):
            for n in range(q):
                if genotype[i] == alphabet[n]:
                    x[i] = n
        return x

    # get allele frequency x_i
    def get_mutant_frequency(pop_at_t):
        genotypes = pop_at_t.keys()
        x = np.zeros(x_length)
        for genotype in genotypes:
            genovalue = get_genotype_value(genotype)
            for i in range(seq_length):
                if genovalue[i] != 0:
                    x[i] += pop_at_t[genotype]
        x_frequencies = x / float(N)
        return x_frequencies

    # calculate single an_del pair allele frequency (binary case)
    def get_allele_frequency(pop_at_t):
        genotypes = pop_at_t.keys()
        x = np.zeros(x_length)             # single allele frequency
        xx = np.zeros([x_length,x_length]) # pair allele frequency
        for genotype in genotypes:
            genovalue = get_genotype_value(genotype)
            # selection part
            for i in range(seq_length):
                if genovalue[i] != 0:
                    x[i] += pop_at_t[genotype]
                    for j in range(int(i+1), seq_length):
                        if genovalue[j] != 0:
                            xx[i,j] += pop_at_t[genotype]
                            xx[j,i] += pop_at_t[genotype]
        x_frequencies = x / float(N)
        xx_frequencies = xx / float(N)
        return x_frequencies,xx_frequencies

    # diffusion matrix C
    def diffusion_matrix_at_t(pop_at_t):
        C = np.zeros([x_length,x_length])
        x_frequencies, xx_frequencies = get_allele_frequency(pop_at_t)
        for i in range(x_length):
            C[i,i] = x_frequencies[i] - x_frequencies[i] * x_frequencies[i]
            for j in range(i+1 ,x_length):
                C[i,j] = xx_frequencies[i,j] - x_frequencies[i] * x_frequencies[j]
                C[j,i] = xx_frequencies[i,j] - x_frequencies[i] * x_frequencies[j]
        return C

    # get denominator
    def get_first_term(history):
        C_int = np.zeros([x_length,x_length])
        for k in range(n_gen-1):
            pop_at_t = history[k] #modify
            C_int +=  diffusion_matrix_at_t(pop_at_t)
        I = np.identity(x_length)
        denominator = np.mat(C_int + I * gamma)
        return np.linalg.inv(denominator)# get the inverse matrix

    def get_flux(pop_at_t):
        x1 = get_mutant_frequency(pop_at_t)
        flux = np.zeros(x_length)
        for i in range(seq_length):
            flux[i] = mu * ( 1 - 2 * x1[i])
        return flux

    def get_second_term(history):
        numerator = np.zeros([x_length,1])
        flux = np.zeros(x_length)
        pop_at_K = history[n_gen]
        pop_at_0 = history[0]
        x_tK = get_mutant_frequency(pop_at_K)
        x_t0 = get_mutant_frequency(pop_at_0)
        for k in range(n_gen-1):
            pop_at_t = history[k]
            flux += get_flux(pop_at_t)
        nu = x_tK - x_t0 - flux
        for i in range(len(nu)):
            numerator[i,0] = nu[i]
        return numerator

    #run mpl
    history = {}
    file = open("./%s/%s.txt"%(SIM_DIR,xfile),'r+');#modify
    history = eval(file.read())
    file.close();
    c_all = np.mat(get_first_term(history))*np.mat(get_second_term(history))
    np.savetxt("./%s/%s.dat"%(SIM_DIR,yfile),c_all)

    if output == True:
        print('Calculation completes')
        print('inferred beneficial selection coefficients are:')
        print(c_all[0:n_ben])
        print('inferred neutral selection coefficients are:')
        print(c_all[n_ben:n_ben+n_neu])
        print('inferred deleterious selection coefficients are:')
        print(c_all[n_ben+n_neu:seq_length])
        print('inferred trait coefficients are:')
        print(c_all[seq_length:])
