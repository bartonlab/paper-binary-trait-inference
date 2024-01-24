import numpy as np
try:
    import itertools.izip as zip
except ImportError:
    import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import time as time_module
import re

# GitHub
HIV_DIR = 'data/HIV'
SIM_DIR = 'data/simulation'
FIG_DIR = 'figures'

def read_file(name):
    trait = []
    p = open(SIM_DIR+'/jobs/'+name,'r')
    maxlen = 0
    maxnum = 0
    for line in p:
        temp = re.findall(r'-?\d+\.?\d*e?-?\d*?', line)
        data = [float(item) for item in temp]
        trait.append(data)
        #print(data)
        if len(data) > maxlen:
            maxlen = len(data)
        maxnum += 1
    arr = np.zeros((maxnum, maxlen))*np.nan
    for cnt in np.arange(maxnum):
        arr[cnt, 0:len(trait[cnt])] = np.array(trait[cnt])
    p.close()
    for i in range(len(trait)):
        for j in range(len(trait[i])):
            trait[i][j] = int(trait[i][j])
    return trait

def simulate(**pdata):
    """
    Example evolutionary trajectory for a 50-site binary system
    """

    # unpack passed data
    alphabet = pdata['alphabet']
    xfile    = pdata['xfile']
    output   = pdata['output']

    n_gen         = pdata['n_gen']
    N             = pdata['N']
    mu            = pdata['mu']
    r_rate        = pdata['r']
    n_ben         = pdata['n_ben']
    n_neu         = pdata['n_neu']
    n_del         = pdata['n_del']
    s_ben         = pdata['s_ben']
    s_neu         = pdata['s_neu']
    s_del         = pdata['s_del']
    s_pol         = pdata['s_pol']
    escape_group  = pdata['escape_group']

    q     = len(alphabet)
    ne = len(escape_group)

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
        for n in range(ne):
            poly_n = 0;
            for j in escape_group[n]:
                poly_n += abs(geno_value[j] - alphabet.in_delex('A'));
            if poly_n != 0:
                fitness[genotype] += s_pol;
        return fitness[genotype]

    # get fitness of new genotype (alphabet format)
    def get_fitness_alpha(genotype):
        fitness = 1.0;
        #alphabet format
        for i in range(n_ben):
            if genotype[i] != "A":
                fitness += s_ben
        for i in range(n_del):
            if genotype[-(i+1)] != "A":
                fitness += s_del
        for n in range(ne):
            for j in escape_group[n]:
                if genotype[j] != "A":
                    fitness += s_pol
                    break
        return fitness

    def get_recombination(genotype1,genotype2):
        #choose one possible mutation site
        recombination_point = np.random.randint(seq_length-1) + 1;
        # get two offspring genotypes
        genotype_off = genotype1[:recombination_point] + genotype2[recombination_point:]
        return genotype_off

    def recombination_event(genotype,genotype_ran,pop):
        if pop[genotype] > 1:
            pop[genotype] -= 1
            if pop[genotype] == 0:
                del pop[genotype]

            new_genotype = get_recombination(genotype,genotype_ran)
            if new_genotype in pop:
                pop[new_genotype] += 1
            else:
                pop[new_genotype] = 1
        return pop

    # create all recombinations that occur in a single generation
    def recombination_step(pop):
        genotypes = list(pop.keys())
        numbers = list(pop.values())
        weights = [float(n) / sum(numbers) for n in numbers]
        for genotype in genotypes:
            n = pop[genotype]
            # calculate the likelihood to recombine
            # recombination rate per locus r,  P = (1 - (1-r)^(L - 1)) = r(L-1)
            total_rec = r_rate*(seq_length - 1)
            nRec = np.random.binomial(n, total_rec)
            for j in range(nRec):
                genotype_ran = np.random.choice(genotypes, p=weights)
                recombination_event(genotype,genotype_ran,pop);
        return pop

    # for different genotypes, they have different mutation probablity
    # this total mutation value represents the likelihood for one genotype to mutate
    # if there is only 2 alleles an_del only one mutation rate, total_mu = mutation rate * sequence length
    # def get_mu(genotype):
    #     total_mu = 0
    #     geno_value = get_genotype_value(genotype)
    #     for i in range(seq_length):
    #         a = int(geno_value[i])
    #         mu_i = sum(muMatrix[a]) # calculate total mutation rate in site i
    #         total_mu += mu_i
    #     return total_mu

    # take a supplied genotype an_del mutate a site at random.
    # def get_mutant(genotype):
    #    #choose one possible mutation site
    #    site = np.random.randint(seq_length);
    #    #choose possible mutation base
    #    geno_value = get_genotype_value(genotype);
    #    a = int(geno_value[site]);
    #    mu_fre = muMatrix[a];
    #    mu_tot = sum(mu_fre);
    #    mu_f = [x / mu_tot for x in mu_fre];
    #    # choose the mutation allele according to mutation rate (for multiple allele case)
    #    mutation = np.random.choice(alphabet, p=mu_f);
    #    # get new mutation sequence
    #    new_genotype = genotype[:site] + mutation[0] + genotype[site+1:];
    #    return new_genotype

    def get_mutant(genotype):
        #choose one possible mutation site
        site = np.random.randint(seq_length);
        # mutate (binary case, from WT to mutant or vice)
        mutation = ['A', 'T']
        mutation.remove(genotype[site])
        # get new mutation sequence
        new_genotype = genotype[:site] + mutation[0] + genotype[site+1:];
        return new_genotype

    # check if the mutant already exists in the population.
    #If it does, increment this mutant genotype;
    #If it doesn't create a new genotype of count 1.
    # If a mutation event creates a new genotype, calculate its fitness.
    def mutation_event(genotype,pop):
        if pop[genotype] > 1:
            pop[genotype] -= 1
            if pop[genotype] == 0:
                del pop[genotype]

            new_genotype = get_mutant(genotype)
            if new_genotype in pop:
                pop[new_genotype] += 1
            else:
                pop[new_genotype] = 1
        return pop

    # create all the mutations that occur in a single generation
    def mutation_step(pop):
        genotypes = list(pop.keys())
        for genotype in genotypes:
            n = pop[genotype]
            # calculate the likelihood to mutate
            total_mu =  mu * seq_length # for binary case
            # total_mu = get_mu(genotype) # for multiple case (mutation rate is different for alleles)
            nMut = np.random.binomial(n, total_mu)
            for j in range(nMut):
                mutation_event(genotype,pop);
        return pop

    # genetic drift
    def offspring_step(pop):
        genotypes = list(pop.keys())
        r = []
        for genotype in genotypes:
            numbers = pop[genotype]
            fitness = get_fitness_alpha(genotype)
            r.append(numbers * fitness)
        weights = [x / sum(r) for x in r]
        pop_size_t = np.sum([pop[i] for i in genotypes])
        counts = list(np.random.multinomial(pop_size_t, weights)) # genetic drift
        for (genotype, count) in zip(genotypes, counts):
            if (count > 0):
                pop[genotype] = count
            else:
                del pop[genotype]
        return pop,fitness

    def simulate(pop,history):
        clone_pop = dict(pop)
        history.append(clone_pop)
        for t in range(n_gen):
            recombination_step(pop)
            mutation_step(pop)
            offspring_step(pop)
            clone_pop = dict(pop)
            history.append(clone_pop)
        return history

    # transfer output from alphabet to number
    def get_sequence(genotype):
        escape_states = []
        for i in range(len(genotype)):
            for k in range(q):
                if genotype[i] == alphabet[k]:
                    escape_states.append(str(k))
        return escape_states

    ############################################################################
    ############################## Simulate ####################################
    pop = {}
    pop["AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"] = N
    history = [];
    simulate(pop,history)

    # write the output file - dat format
    f = open("./%s/jobs/sequences/example-%s.dat"%(SIM_DIR,xfile),'w')
    for i in range(len(history)):
        pop_at_t = history[i]
        genotypes = pop_at_t.keys()
        for genotype in genotypes:
            time = i
            counts = pop_at_t[genotype]
            sequence = get_sequence(genotype)
            f.write('%d\t%d\t' % (time,counts))
            for j in range(len(sequence)):
                f.write(' %s' % (' '.join(sequence[j])))
            f.write('\n')
    f.close()


    if output == True:
        print('Simulation completes')
        print('beginning with wild type, sequence has %d types of alleles, totally run %d generations' %(q,n_gen))
        print('the length of the sequence is %d, with %d beneficial sites (%s) and %d deleterious sites（%s)'%(seq_length,n_ben,s_ben,n_del,s_del))
        print('containing %d trait groups with trait coefficients equal to %s, they are '%(ne,s_pol))
        print(escape_group)

def run_mpl_binary(**pdata):
    """
    Use mpl to calculate selection coefficients and trait coefficients (binary case)
    Evolutionary force : selection, recombination and mutation
    """
    # unpack passed data
    n_gen    = pdata['n_gen']
    N        = pdata['N']
    alphabet = pdata['alphabet']
    xfile    = pdata['xfile']
    save_cov = pdata['save_cov']
    output   = pdata['output']

    mu      = pdata['mu']
    r_rate  = pdata['r']
    n_ben   = pdata['n_ben']
    n_neu   = pdata['n_neu']
    n_del   = pdata['n_del']
    s_ben   = pdata['s_ben']
    s_neu   = pdata['s_neu']
    s_del   = pdata['s_del']
    s_pol   = pdata['s_pol']
    gamma_s = pdata['gamma']
    gamma_t = gamma_s/10

    escape_group  = read_file('traitsite/traitsite-%s.dat'%(xfile[0]))
    trait_dis     = read_file('traitdis/traitdis-%s.dat'%(xfile[0]))

    ne            = len(escape_group)
    seq_length    = n_ben+n_neu+n_del
    x_length      = seq_length+ne

    # start_time = time_module.time()
    ############################################################################
    ############################## Function ####################################
    # loading data from dat file
    def getSequenceT(data,escape_group,time):
        # find index for time t
        idx  = data.T[0]==time
        nVec = data[idx].T[1]
        sVec = data[idx].T[2:].T

        ne   = len(escape_group)
        indices = [i for i, val in enumerate(idx) if val]
        if ne > 0: # the patient contains escape group
            eVec = np.zeros((len(sVec),ne), dtype=int)
            sWT  = np.zeros(seq_length)
            for i in range(len(sVec)):
                for n in range(ne):
                    sWT_n  = [sWT[j] for j in escape_group[n]]
                    eVec_n = [sVec[i][j] for j in escape_group[n]]
                    if sWT_n[:] != eVec_n[:]:
                        eVec[i,n] = 1
        return sVec,nVec,eVec

    # calculate single and pair allele frequency (binary case)
    def get_allele_frequency(sVec,nVec,eVec):
        x  = np.zeros(x_length)          # single allele frequency
        xx = np.zeros((x_length,x_length))  # pair allele frequency
        pop_size_t = np.sum(nVec)
        for i in range(seq_length):
            x[i] = np.sum([sVec[k][i] * nVec[k] for k in range(len(sVec))]) / pop_size_t
            for j in range(int(i+1), seq_length):
                xx[i,j] = np.sum([sVec[k][i] * sVec[k][j] * nVec[k] for k in range(len(sVec))]) / pop_size_t
                xx[j,i] = np.sum([sVec[k][i] * sVec[k][j] * nVec[k] for k in range(len(sVec))]) / pop_size_t
        for n in range(ne):
            aa      = x_length-ne+n
            x[aa] = np.sum([eVec[k][n] * nVec[k] for k in range(len(sVec))]) / pop_size_t
            for j in range(seq_length):
                xx[aa,j] = np.sum([sVec[k][j] * eVec[k][n] * nVec[k] for k in range(len(sVec))]) / pop_size_t
                xx[j,aa] = np.sum([sVec[k][j] * eVec[k][n] * nVec[k] for k in range(len(sVec))]) / pop_size_t
            for m in range(int(n+1), ne):
                bb        = x_length-ne+m
                xx[aa,bb] = np.sum([eVec[k][n] * eVec[k][m] * nVec[k] for k in range(len(sVec))]) / pop_size_t
                xx[bb,aa] = np.sum([eVec[k][n] * eVec[k][m] * nVec[k] for k in range(len(sVec))]) / pop_size_t
        return x,xx

    # calculate escape frequency (binary case)
    def get_escape_fre_term(sVec,nVec):
        ex  = np.zeros((ne,seq_length))
        pop_size_t = np.sum(nVec)
        for k in range(len(sVec)):
            for n in range(ne):
                n_mutations = 0
                for nn in escape_group[n]:
                    if sVec[k][nn] != 0:
                        n_mutations += 1
                        site = nn
                if n_mutations == 1:
                    ex[n,site] += nVec[k]
        ex[:,:] = ex[:,:] / pop_size_t
        return ex

    # calculate frequencies for recombination part (binary case)
    def get_p_k(sVec,nVec,escape_group_n):
        p_mut_k   = np.zeros((3,len(escape_group_n)-1))
        p_wt      = 0
        sWT       = np.zeros(seq_length)
        pop_size_t = np.sum(nVec)
        sWT_n  = [sWT[i] for i in escape_group_n]
        for k in range(len(sVec)):
            sVec_n = [sVec[k][i] for i in escape_group_n]
            # no mutation within the trait group
            if sWT_n == sVec_n:
                p_wt += nVec[k]

            for nn in range(len(escape_group_n)-1):
                k_bp = nn + 1
                # containing mutation before and after break point k
                if sWT_n[:k_bp] != sVec_n[:k_bp] and sWT_n[k_bp:] != sVec_n[k_bp:]:
                    p_mut_k[0,nn] += nVec[k]
                # MT before break point k and WT after break point k
                if sWT_n[:k_bp] != sVec_n[:k_bp] and sWT_n[k_bp:] == sVec_n[k_bp:]:
                    p_mut_k[1,nn] += nVec[k]
                # WT before break point k and MT after break point k
                if sWT_n[:k_bp] == sVec_n[:k_bp] and sWT_n[k_bp:] != sVec_n[k_bp:]:
                    p_mut_k[2,nn] += nVec[k]

        p_wt    = p_wt / pop_size_t
        p_mut_k = p_mut_k / pop_size_t

        return p_wt,p_mut_k

    # calculate recombination flux term (binary_case)
    def get_recombination_flux(escape_group,trait_dis):
        flux = np.zeros(x_length)
        for n in range(ne):
            escape_group_n = escape_group[n]
            p_wt,p_mut_k = get_p_k(sVec,nVec,escape_group_n)

            fluxIn  = 0
            fluxOut = 0

            for nn in range(len(p_mut_k[0])):
                fluxIn  += trait_dis[n][nn] * p_wt*p_mut_k[0,nn]
                fluxOut += trait_dis[n][nn] * p_mut_k[1,nn]*p_mut_k[2,nn]
            flux[x_length-ne+n] += r_rate * (fluxIn - fluxOut)

        return flux

    # calculate mutation flux term (binary_case)
    def get_mu_flux(x,ex):
        flux = np.zeros(x_length)
        for i in range(seq_length):
            flux[i] = mu * ( 1 - 2 * x[i]);
        for n in range(ne):
            for nn in escape_group[n]:
                flux[x_length-ne+n] += mu * (1 - x[x_length-ne+n] - ex[n,nn] )
        return flux

    # calculate diffusion matrix C
    def diffusion_matrix_at_t(x,xx,x_length):
        C = np.zeros((x_length,x_length))
        for i in range(x_length):
            C[i,i] = x[i] - x[i] * x[i]
            for j in range(int(i+1) ,x_length):
                C[i,j] = xx[i,j] - x[i] * x[j]
                C[j,i] = xx[i,j] - x[i] * x[j]
        return C

    ############################################################################
    ####################### Inference (binary case) ############################
    # obtain raw data
    data      = np.loadtxt("%s/jobs/sequences/example-%s.dat"%(SIM_DIR,xfile))
    times     = np.unique(data.T[0]) # get all time point

    totalCov  = np.zeros([x_length,x_length])
    bayesian  = np.zeros([x_length,x_length])
    totalflux = np.zeros(x_length)
    totalRec  = np.zeros(x_length)

    for t in range(len((times))-1):
        time = times[t]
        dt   = times[t+1] - times[t]
        sVec,nVec,eVec = getSequenceT(data,escape_group,time)
        x,xx   = get_allele_frequency(sVec,nVec,eVec)
        ex     = get_escape_fre_term(sVec,nVec)
        totalflux += dt * get_mu_flux(x,ex) # mutation part
        totalRec  += dt * get_recombination_flux(escape_group,trait_dis)# recombination part
        totalCov  += dt * diffusion_matrix_at_t(x,xx,x_length)
        if time == 0:
            x_t0 = x

    # regularization part (using different γ for individual locus and trait group part)
    for i in range(x_length-ne):
        bayesian[i, i] += gamma_s
    for n in range(ne):
        ii = x_length - ne + n
        bayesian[ii, ii] += gamma_t

    LHS_av = totalCov + bayesian
    RHS_av = np.zeros((x_length,1))

    sVec,nVec,eVec = getSequenceT(data,escape_group,times[-1])
    x,xx   = get_allele_frequency(sVec,nVec,eVec)
    x_tK   = x

    for i in range(x_length):
        RHS_av[i,0] =  x_tK[i] - x_t0[i] - totalflux[i] - totalRec[i]

    solution_const_av = np.linalg.solve(LHS_av, RHS_av)
    sc = solution_const_av.reshape(-1)
    np.savetxt("%s/jobs/output/sc-%s-python.dat"%(SIM_DIR,xfile),sc)

    # save total covariance
    if save_cov == True:
        np.savetxt("./%s/example_covariance.dat"%SIM_DIR,totalCov)
    # end_time = time_module.time()
    # print(f"Execution time for simulation (binary case): {end_time - start_time} seconds")

    # output
    if output == True:
        print('Calculation completes')
        print('inferred beneficial selection coefficients are:')
        print(sc[0:n_ben])
        print('inferred neutral selection coefficients are:')
        print(sc[n_ben:n_ben+n_neu])
        print('inferred deleterious selection coefficients are:')
        print(sc[n_ben+n_neu:seq_length])
        print('inferred trait coefficients are:')
        print(sc[seq_length:])


def py2c(**pdata):

    """
    Convert a trajectory into plain text to save the results.
    """

    # unpack passed data

    t0       = pdata['t0']
    tk       = pdata['T']
    ns_vals  = pdata['ns']
    dt_vals  = pdata['dt']
    xfile    = pdata['xfile']
    alphabet = pdata['alphabet']

    rng = np.random.RandomState()

    # write the results
    for i in range(len(ns_vals)):
        ns      = ns_vals[i]
        for j in range(len(dt_vals)):
            dt = dt_vals[j]
            f  = open('%s/jobs/sequences/nsdt/%s_ns%d_dt%d.dat' % (SIM_DIR, xfile, ns, dt), 'w')
            if dt == 1:
                data  = np.loadtxt("%s/jobs/sequences/%s_ns1000_dt1.dat"%(SIM_DIR,xfile))
                for tt in range(t0, tk+1, dt):
                    idx  = data.T[0]==tt
                    nVec_t = data[idx].T[1]
                    sVec_t = data[idx].T[2:].T

                    iVec = np.zeros(int(np.sum(nVec_t)))
                    ct   = 0
                    for k in range(len(nVec_t)):
                        iVec[ct:int(ct+nVec_t[k])] = k
                        ct += int(nVec_t[k])
                    iSample = rng.choice(iVec, ns, replace=False)
                    for k in range(len(nVec_t)):
                        nSample = np.sum(iSample==k)
                        if nSample>0:
                            f.write('%d\t%d\t%s\n' %(tt, nSample, ' '.join([str(int(kk)) for kk in sVec_t[k]])))

            else:
                data = np.loadtxt('%s/jobs/sequences/nsdt/%s_ns%d_dt1.dat'%(SIM_DIR, xfile,ns))
                for tt in range(t0, tk+1, dt):
                    idx  = data.T[0]==tt
                    nVec_t = data[idx].T[1]
                    sVec_t = data[idx].T[2:].T
                    for k in range(len(nVec_t)):
                        f.write('%d\t%d\t%s\n' %(tt, nVec_t[k], ' '.join([str(int(kk)) for kk in sVec_t[k]])))
            f.close()

def simulate_multiple(**pdata):
    """
    Example evolutionary trajectory for a 50-site multiple system
    """

    # unpack passed data
    alphabet = pdata['alphabet']
    xfile    = pdata['xfile']
    output   = pdata['output']

    n_gen         = pdata['n_gen']
    N             = pdata['N']
    mu            = pdata['mu']
    r_rate        = pdata['r']
    n_ben         = pdata['n_ben']
    n_neu         = pdata['n_neu']
    n_del         = pdata['n_del']
    s_ben         = pdata['s_ben']
    s_neu         = pdata['s_neu']
    s_del         = pdata['s_del']
    s_pol         = pdata['s_pol']
    escape_group  = pdata['escape_group']

    q     = len(alphabet)
    ne = len(escape_group)

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
        for n in range(ne):
            poly_n = 0;
            for j in escape_group[n]:
                poly_n += abs(geno_value[j] - alphabet.in_delex('A'));
            if poly_n != 0:
                fitness[genotype] += s_pol;
        return fitness[genotype]

    # get fitness of new genotype (alphabet format)
    def get_fitness_alpha(genotype):
        fitness = 1.0;
        #alphabet format
        for i in range(n_ben):
            if genotype[i] != "A":
                fitness += s_ben
        for i in range(n_del):
            if genotype[-(i+1)] != "A":
                fitness += s_del
        for n in range(ne):
            for j in escape_group[n]:
                if genotype[j] != "A":
                    fitness += s_pol
                    break
        return fitness

    def get_recombination(genotype1,genotype2):
        #choose one possible mutation site
        recombination_point = np.random.randint(seq_length-1) + 1;
        # get two offspring genotypes
        genotype_off = genotype1[:recombination_point] + genotype2[recombination_point:]
        return genotype_off

    def recombination_event(genotype,genotype_ran,pop):
        if pop[genotype] > 1:
            pop[genotype] -= 1
            if pop[genotype] == 0:
                del pop[genotype]

            new_genotype = get_recombination(genotype,genotype_ran)
            if new_genotype in pop:
                pop[new_genotype] += 1
            else:
                pop[new_genotype] = 1
        return pop

    # create all recombinations that occur in a single generation
    def recombination_step(pop):
        genotypes = list(pop.keys())
        numbers = list(pop.values())
        weights = [float(n) / sum(numbers) for n in numbers]
        for genotype in genotypes:
            n = pop[genotype]
            # calculate the likelihood to recombine
            # recombination rate per locus r,  P = (1 - (1-r)^(L - 1)) = r(L-1)
            total_rec = r_rate*(seq_length - 1)
            nRec = np.random.binomial(n, total_rec)
            for j in range(nRec):
                genotype_ran = np.random.choice(genotypes, p=weights)
                recombination_event(genotype,genotype_ran,pop);
        return pop

    # for different genotypes, they have different mutation probablity
    # this total mutation value represents the likelihood for one genotype to mutate
    # if there is only 2 alleles an_del only one mutation rate, total_mu = mutation rate * sequence length
    # def get_mu(genotype):
    #     total_mu = 0
    #     geno_value = get_genotype_value(genotype)
    #     for i in range(seq_length):
    #         a = int(geno_value[i])
    #         mu_i = sum(muMatrix[a]) # calculate total mutation rate in site i
    #         total_mu += mu_i
    #     return total_mu

    # take a supplied genotype an_del mutate a site at random.
    # def get_mutant(genotype):
    #    #choose one possible mutation site
    #    site = np.random.randint(seq_length);
    #    #choose possible mutation base
    #    geno_value = get_genotype_value(genotype);
    #    a = int(geno_value[site]);
    #    mu_fre = muMatrix[a];
    #    mu_tot = sum(mu_fre);
    #    mu_f = [x / mu_tot for x in mu_fre];
    #    # choose the mutation allele according to mutation rate (for multiple allele case)
    #    mutation = np.random.choice(alphabet, p=mu_f);
    #    # get new mutation sequence
    #    new_genotype = genotype[:site] + mutation[0] + genotype[site+1:];
    #    return new_genotype

    def get_mutant(genotype):
        #choose one possible mutation site
        site = np.random.randint(seq_length);
        # mutate (binary case, from WT to mutant or vice)
        mutation = ['A', 'T']
        mutation.remove(genotype[site])
        # get new mutation sequence
        new_genotype = genotype[:site] + mutation[0] + genotype[site+1:];
        return new_genotype

    # check if the mutant already exists in the population.
    #If it does, increment this mutant genotype;
    #If it doesn't create a new genotype of count 1.
    # If a mutation event creates a new genotype, calculate its fitness.
    def mutation_event(genotype,pop):
        if pop[genotype] > 1:
            pop[genotype] -= 1
            if pop[genotype] == 0:
                del pop[genotype]

            new_genotype = get_mutant(genotype)
            if new_genotype in pop:
                pop[new_genotype] += 1
            else:
                pop[new_genotype] = 1
        return pop

    # create all the mutations that occur in a single generation
    def mutation_step(pop):
        genotypes = list(pop.keys())
        for genotype in genotypes:
            n = pop[genotype]
            # calculate the likelihood to mutate
            total_mu =  mu * seq_length # for binary case
            # total_mu = get_mu(genotype) # for multiple case (mutation rate is different for alleles)
            nMut = np.random.binomial(n, total_mu)
            for j in range(nMut):
                mutation_event(genotype,pop);
        return pop

    # genetic drift
    def offspring_step(pop):
        genotypes = list(pop.keys())
        r = []
        for genotype in genotypes:
            numbers = pop[genotype]
            fitness = get_fitness_alpha(genotype)
            r.append(numbers * fitness)
        weights = [x / sum(r) for x in r]
        pop_size_t = np.sum([pop[i] for i in genotypes])
        counts = list(np.random.multinomial(pop_size_t, weights)) # genetic drift
        for (genotype, count) in zip(genotypes, counts):
            if (count > 0):
                pop[genotype] = count
            else:
                del pop[genotype]
        return pop,fitness

    def simulate(pop,history):
        clone_pop = dict(pop)
        history.append(clone_pop)
        for t in range(n_gen):
            recombination_step(pop)
            mutation_step(pop)
            offspring_step(pop)
            clone_pop = dict(pop)
            history.append(clone_pop)
        return history

    # transfer output from alphabet to number
    def get_sequence(genotype):
        escape_states = []
        for i in range(len(genotype)):
            for k in range(q):
                if genotype[i] == alphabet[k]:
                    escape_states.append(str(k))
        return escape_states

    ############################################################################
    ############################## Simulate ####################################
    pop = {}
    pop["AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"] = N
    history = [];
    simulate(pop,history)

    # write the output file - dat format
    f = open("./%s/%s.dat"%(SIM_DIR,xfile),'w')
    for i in range(len(history)):
        pop_at_t = history[i]
        genotypes = pop_at_t.keys()
        for genotype in genotypes:
            time = i
            counts = pop_at_t[genotype]
            sequence = get_sequence(genotype)
            f.write('%d\t%d\t' % (time,counts))
            for j in range(len(sequence)):
                f.write(' %s' % (' '.join(sequence[j])))
            f.write('\n')
    f.close()


    if output == True:
        print('Simulation completes')
        print('beginning with wild type, sequence has %d types of alleles, totally run %d generations' %(q,n_gen))
        print('the length of the sequence is %d, with %d beneficial sites (%s) and %d deleterious sites（%s)'%(seq_length,n_ben,s_ben,n_del,s_del))
        print('containing %d trait groups with trait coefficients equal to %s, they are '%(ne,s_pol))
        print(escape_group)

def run_mpl_multiple(**pdata):

    """
    Use mpl to calculate selection coefficients and trait coefficients (binary case)
    """

    # unpack passed data

    n_gen    = pdata['n_gen']
    N        = pdata['N']
    escape_group  = pdata['po_site']
    alphabet = pdata['alphabet']
    xfile    = pdata['xfile']
    yfile    = pdata['yfile']
    output   = pdata['output']

    mu    = pdata['mu']
    n_ben = pdata['n_ben']
    n_neu = pdata['n_neu']
    n_del = pdata['n_del']
    ne = pdata['ne']
    s_ben = pdata['s_ben']
    s_neu = pdata['s_neu']
    s_del = pdata['s_del']
    s_pol = pdata['s_pol']
    gamma = pdata['gamma']

    q     = len(alphabet)
    seq_length  = n_ben+n_neu+n_del
    x_length = seq_length*q + ne  #binary case

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
        p = np.zeros(ne)
        for n in range(ne):
            for j in escape_group[n]:
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
            for n in range(ne):
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
                for n in range(ne):
                    if polyvalue[n] == 1:
                        bb = seq_length * q + n
                        xx[aa,bb] += pop_at_t[genotype]
                        xx[bb,aa] += pop_at_t[genotype]
            # trait part
            for n in range(ne):
                aa = seq_length * q + n
                if polyvalue[n] == 1:
                    x[aa] += pop_at_t[genotype]
                    for m in range(ne):
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
    def get_mu_frequency(pop_at_t,escape_group):
        genotypes = pop_at_t.keys()
        x_mu = np.zeros([seq_length,q])
        for genotype in genotypes:
            genovalue = get_genotype_value(genotype)
            n_mutations = 0
            for j in escape_group:
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
        for n in range(ne):
            x2 = get_mu_frequency(pop_at_t,escape_group[n])
            for j in escape_group[n]:
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
    sc_all = np.zeros(seq_length*(q-1)+ne)
    for j in range(seq_length):
        for a in range(q-1):
            index = j * (q - 1) + a
            sc_all[index] = c_all[j*q+a+1] - c_all[j*q]
    for n in range(ne):
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
