import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import time as time_module
import re

# GitHub
HIV_DIR = 'data/HIV'
SIM_DIR = 'data/simulation'
FIG_DIR = 'figures'

def read_file(path,name):
    trait = []
    p = open(SIM_DIR+'/'+path+'/'+name,'r')
    maxlen = 0
    maxnum = 0
    for line in p:
        temp = re.findall(r'-?\d+\.?\d*e?-?\d*?', line)
        data = [float(item) for item in temp]
        trait.append(data)
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
    xpath    = pdata['xpath']
    xfile    = pdata['xfile']
    example  = pdata['example']

    n_gen         = pdata['n_gen']
    pop_size      = pdata['N']
    mu            = pdata['mu']
    r_rate        = pdata['r']
    n_ben         = pdata['n_ben']
    n_neu         = pdata['n_neu']
    n_del         = pdata['n_del']
    s_ben         = pdata['s_ben']
    s_del         = pdata['s_del']
    s_tra         = pdata['s_tra']
    escape_group  = pdata['escape_group']

    q     = len(alphabet)
    ne = len(escape_group)

    seq_length = n_ben+n_neu+n_del

    ############################################################################
    ############################## FUNCTIONS ###################################
    # get fitness of new genotype (alphabet format)
    def get_fitness_alpha(genotype):
        fitness = 1.0
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
                    fitness += s_tra
                    break
        return fitness

    def get_recombination(genotype1,genotype2):
        #choose one possible mutation site
        recombination_point = np.random.randint(seq_length-1) + 1
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
                recombination_event(genotype,genotype_ran,pop)
        return pop

    def get_mutant(genotype):
        #choose one possible mutation site
        site = np.random.randint(seq_length)
        # mutate (binary case, from WT to mutant or vice)
        mutation = ['A', 'T']
        mutation.remove(genotype[site])
        # get new mutation sequence
        new_genotype = genotype[:site] + mutation[0] + genotype[site+1:]
        return new_genotype

    # check if the mutant already exists in the population.
    #If it does, increment this mutant genotype
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
                mutation_event(genotype,pop)
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
    pop["AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"] = pop_size
    history = []
    simulate(pop,history)

    # write the output file - dat format
    f = open("./%s/%s/%s.dat"%(SIM_DIR,xpath,xfile),'w')
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

    if example == True:
        print('Simulation completes')
        print('beginning with wild type, sequence has %d types of alleles, totally run %d generations' %(q,n_gen))
        print('the length of the sequence is %d, with %d beneficial sites (%s) and %d deleterious sites（%s)'%(seq_length,n_ben,s_ben,n_del,s_del))
        print('containing %d trait groups with trait coefficients equal to %s, they are '%(ne,s_tra))
        print(escape_group)


def simulate_tv(**pdata):
    """
    Example evolutionary trajectory for a 20-site system
    """

    # unpack passed data
    alphabet      = pdata['alphabet']       # ['A','T']
    xpath         = pdata['xpath']          # 'time-varying'
    xfile         = pdata['xfile']          # '0_ns1000_dt1'
    n_gen         = pdata['n_gen']
    pop_size      = pdata['N']              # 1000
    mu            = pdata['mu']             # 2e-4
    r_rate        = pdata['r']              # 2e-4
    n_ben         = pdata['n_ben']          # 2
    n_neu         = pdata['n_neu']          # 6
    n_del         = pdata['n_del']          # 2
    s_ben         = pdata['s_ben']          # 0.02
    s_del         = pdata['s_del']          # -0.02
    s_tra         = pdata['s_tra']
    escape_group  = pdata['escape_group']
    inital_state  = pdata['inital_state']   # 4

    q  = len(alphabet)
    ne = len(escape_group)

    seq_length = n_ben+n_neu+n_del

    ############################################################################
    ############################## function ####################################
    # get fitness of new genotype
    def get_fitness_alpha(genotype,time):
        fitness = 1.0
    
        # individual locus
        for i in range(n_ben):
            if genotype[i] != "A": # beneficial mutation
                fitness += s_ben
        for i in range(n_del):
            if genotype[-(i+1)] != "A": # deleterious mutation
                fitness += s_del
        
        # binary trait
        for n in range(ne):
            for nn in escape_group[n]:
                if genotype[nn] != "A":
                    fitness += s_tra[time]
                    break
        return fitness

    def get_recombination(genotype1,genotype2):
        #choose one possible mutation site
        recombination_point = np.random.randint(seq_length-1) + 1
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
            nflux_rec = np.random.binomial(n, total_rec)
            for j in range(nflux_rec):
                genotype_ran = np.random.choice(genotypes, p=weights)
                recombination_event(genotype,genotype_ran,pop)
        return pop
    
    # for different genotypes, they have different mutation probablity
    # this total mutation value represents the likelihood for one genotype to mutate
    # if there is only 2 alleles and only one mutation rate, total_mu = mutation rate * sequence length
    # take a supplied genotype and mutate a site at random.
    def get_mutant(genotype): #binary case
        #choose one possible mutation site
        site = np.random.randint(seq_length)
        # mutate (binary case, from WT to mutant or vice)
        mutation = list(alphabet)
        mutation.remove(genotype[site])
        # get new mutation sequence
        new_genotype = genotype[:site] + mutation[0] + genotype[site+1:]
        return new_genotype

    # check if the mutant already exists in the population.
    #If it does, increment this mutant genotype
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
            total_mu = seq_length * mu # for binary case
            nMut = np.random.binomial(n, total_mu)
            for j in range(nMut):
                mutation_event(genotype,pop)
        return pop

    # genetic drift
    def offspring_step(pop,time):
        genotypes = list(pop.keys())
        r = []
        for genotype in genotypes:
            numbers = pop[genotype]
            fitness = get_fitness_alpha(genotype,time)
            r.append(numbers * fitness)
        weights = [x / sum(r) for x in r]
        pop_size_t = np.sum([pop[i] for i in genotypes])
        counts = list(np.random.multinomial(pop_size_t, weights)) # genetic drift
        for (genotype, count) in zip(genotypes, counts):
            if (count > 0):
                pop[genotype] = count
            else:
                del pop[genotype]
        return pop

    # in every generation, it will mutate and then the genetic drift
    # calculate several times to get the evolution trajectory
    # At each step in the simulation, we append to a history object.
    def simulate(pop,history):
        clone_pop = dict(pop)
        history.append(clone_pop)
        for t in range(n_gen):
            recombination_step(pop)
            mutation_step(pop)
            offspring_step(pop,t)
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

    def initial_dis(pop,inital_state,pop_size):
        n_seqs  = int(pop_size/inital_state)
        for ss in range(inital_state):
            sequences = ''
            for i in range(seq_length):
                temp_seq   = np.random.choice(np.arange(0, q), p=[0.8, 0.2])
                allele_i   = alphabet[temp_seq]
                sequences += allele_i
            if ss != inital_state-1:
                if sequences in pop:
                    pop[sequences] += n_seqs
                else:
                    pop[sequences]  = n_seqs
            else:
                if sequences in pop:
                    pop[sequences] += pop_size - (inital_state-1)*n_seqs
                else:
                    pop[sequences]  = pop_size - (inital_state-1)*n_seqs

    ############################################################################
    ############################## Simulate ####################################
    pop = {}
    pop["AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"] = pop_size

    history = []
    simulate(pop,history)

    # write the output file - dat format
    f = open("%s/%s/sequences/%s.dat"%(SIM_DIR,xpath,xfile),'w')

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

def run_mpl_binary(**pdata):
    """
    Use mpl to calculate selection coefficients and trait coefficients (binary case)
    Evolutionary force : selection, recombination and mutation
    """
    # unpack passed data
    xpath    = pdata['xpath']
    xfile    = pdata['xfile']
    save_cov = pdata['save_cov']
    example  = pdata['example']

    mu      = pdata['mu']
    r_rate  = pdata['r']
    n_ben   = pdata['n_ben']
    n_neu   = pdata['n_neu']
    n_del   = pdata['n_del']
    gamma_s = pdata['gamma']
    gamma_t = gamma_s/10

    start_time = time_module.time()
    ############################################################################
    ############################## Function ####################################
    # loading data from dat file
    def getSequenceT(data,escape_group,time):
        # find index for time t
        idx  = data.T[0]==time
        nVec = data[idx].T[1]
        sVec = data[idx].T[2:].T

        ne   = len(escape_group)
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
            flux[i] = mu * ( 1 - 2 * x[i])
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

    if example == True:
        data         = np.loadtxt("%s/%s/example-%s.dat"%(SIM_DIR,xpath,xfile))
        escape_group = read_file(xpath,'traitsite-%s.dat'%(xfile[0]))
        trait_dis    = read_file(xpath,'traitsite-%s.dat'%(xfile[0]))
    else:
        data         = np.loadtxt("%s/%s/sequences/example-%s.dat"%(SIM_DIR,xpath,xfile))
        escape_group = read_file(xpath,'traitsite/traitsite-%s.dat'%(xfile[0]))
        trait_dis    = read_file(xpath,'traitdis/traitdis-%s.dat'%(xfile[0]))

    times      = np.unique(data.T[0]) # get all time point
    ne         = len(escape_group)
    seq_length = n_ben+n_neu+n_del
    x_length   = seq_length+ne

    # Calculate the covariance matrix C and drift verctor (mutation flux and recombination flux)
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
    if example == True:
        np.savetxt("%s/%s/sc-%s-python.dat"%(SIM_DIR,xpath,xfile),sc)
    else:
        np.savetxt("%s/%s/outout_python/sc-%s.dat"%(SIM_DIR,xpath,xfile),sc)

    # save total covariance
    if save_cov == True:
        if example == True:
            np.savetxt("./%s/%s/covariance-%s-python.dat"%(SIM_DIR,xpath,xfile),totalCov)
        else:
            np.savetxt("./%s/%s/covariance_python/covariance-%s.dat"%(SIM_DIR,xpath,xfile),totalCov)
    
    end_time = time_module.time()
    print(f"Execution time for simulation (binary case): {end_time - start_time} seconds")

    # output the inferred coefficients
    if example == True:
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
                    idx    = data.T[0]==tt
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
                    idx    = data.T[0]==tt
                    nVec_t = data[idx].T[1]
                    sVec_t = data[idx].T[2:].T
                    for k in range(len(nVec_t)):
                        f.write('%d\t%d\t%s\n' %(tt, nVec_t[k], ' '.join([str(int(kk)) for kk in sVec_t[k]])))
            f.close()

