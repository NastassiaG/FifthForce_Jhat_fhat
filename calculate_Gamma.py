import numpy as np
from scipy.special import hyp2f1


# Calculating LCDM predictions


def E(z, Omega_m0=0.3111): 
    return np.sqrt(Omega_m0*(1+z)**3+(1-Omega_m0))

def Hprime(z, Omega_m0=0.3111): # derivative of the conformal Hubble parameter wrt ln(a)
    return E(z, Omega_m0 = Omega_m0)/(1+z)-Omega_m0*3*(1+z)**2/(2*E(z, Omega_m0 = Omega_m0))


def matter_growth(z, Omega_m0 = 0.3111):
    A = (-1+Omega_m0)/Omega_m0
    return 1/(1+z)*hyp2f1(1/3,1,11/6,A/(1+z)**3)/hyp2f1(1/3,1,11/6,A)

def fhat_LCDM(z, Omega_m0 = 0.3111, sigma_80 = 0.811):
    
    # Calculating f_LCDM, which is given by dln(D)/dln(a), with D being the matter growth function 
    A = (-1+Omega_m0)/Omega_m0
    term1 = hyp2f1(1/3,1,11/6,A/(1+z)**3)
    term2 = 1/(1+z)**3*6/11*A*hyp2f1(4/3,2,17/6,A/(1+z)**3)
    f_LCDM = (term1+term2)/hyp2f1(1/3,1,11/6,A/(1+z)**3)
    
    # Calculating sigma80
    sigma_8 = sigma_80 * matter_growth(z, Omega_m0 = 0.3111)
    
    return f_LCDM*sigma_8

def Jhat_LCDM(z, Omega_m0 = 0.3111, sigma_80 = 0.811, Jhat_error_rel = 0.2, return_error = False):
    
    sigma_8z = sigma_80*matter_growth(z, Omega_m0 = Omega_m0)
    
    Omega_mz = Omega_m0*(1+z)**3/(Omega_m0*(1+z)**3+(1-Omega_m0))
    Jhat=Omega_mz*sigma_8z
    
    if return_error:
        result = [Jhat, Jhat*Jhat_error_rel]
    else:
        result = Jhat
    
    return result

# Calculate derivative from fhat_reco, and Gamma from fhat_reco and Jhat_reco

def dlnf_dln1pz(fhat_reco, N = 10000, include_cov01 = False):
    z_range = fhat_reco['x']
    dlnf_mean = (1+z_range)*fhat_reco["mean_d1"]/fhat_reco['mean']

    if include_cov01:
        comb_cov = np.block([[fhat_reco['cov'], fhat_reco['cov_01']], [fhat_reco['cov_01'].T, fhat_reco['cov_d1']]])
        fhat_mean01 = np.concatenate((fhat_reco['mean'],fhat_reco['mean_d1']))
        samples_fhat01 = np.random.multivariate_normal(fhat_mean01, comb_cov, size = N)
        samples_fhat=samples_fhat01[:,0:len(z_range)]
        samples_fhat_der=samples_fhat01[:,len(z_range):]
    
    else:
        # Generate N different samples
        samples_fhat = np.random.multivariate_normal(fhat_reco['mean'], fhat_reco['cov'], size = N)
        samples_fhat_der = np.random.multivariate_normal(fhat_reco['mean_d1'], fhat_reco['cov_d1'], size = N)            
        
    dlnfdln1pz_values = np.zeros((N,len(z_range)))
    for i in range(N):
        dlnfdln1pz_values[i] = samples_fhat_der[i]*(1+z_range)/samples_fhat[i]
    dlnf_cov = np.cov(dlnfdln1pz_values.T)  
        
    return [dlnf_mean, dlnf_cov]  

def Gamma_reco(fhat_reco, Jhat_reco, N = 10000, include_cov01 = False, output_dlnz = True): #calculate Gamma and its covariance
    z_range = fhat_reco['x']
    
    H = E(z_range)/(1+z_range)
    H_prime = Hprime(z_range)
    fprime = -(1+z_range)*fhat_reco["mean_d1"]
    
    #calculating Gamma (same as first calculating the samples and then taking the mean?)
    Gamma = 2/3*np.array(fhat_reco["mean"]/Jhat_reco["mean"])*(H_prime/H+1+fprime/fhat_reco['mean'])
    
    # calculating the covariance
    Gamma_values = np.zeros((N,len(z_range)))
    samples_Jhat = np.random.multivariate_normal(Jhat_reco["mean"], Jhat_reco['cov'], size = N)

    if include_cov01:
        comb_cov = np.block([[fhat_reco['cov'], fhat_reco['cov_01']], [fhat_reco['cov_01'].T, fhat_reco['cov_d1']]])
        fhat_mean01 = np.concatenate((fhat_reco['mean'], fhat_reco['mean_d1']))
        samples_fhat01 = np.random.multivariate_normal(fhat_mean01, comb_cov, size = N)
        samples_fhat=samples_fhat01[:,0:len(z_range)]
        samples_fhat_der=samples_fhat01[:,len(z_range):]
    
    else:
        # Generate N different samples for one mock data set
        samples_fhat = np.random.multivariate_normal(fhat_reco['mean'], fhat_reco['cov'], size = N)
        samples_fhat_der = np.random.multivariate_normal(fhat_reco['mean_d1'], fhat_reco['cov_d1'], size = N)            
    
    for i in range(N):
        Gamma_values[i] = 2/3*np.array(samples_fhat[i]/samples_Jhat[i])*(H_prime/H+1-(1+z_range)*np.array(samples_fhat_der[i])/samples_fhat[i])
    Gamma_cov = np.cov(Gamma_values.T)
        
    if output_dlnz:
        dlnf_mean = -fprime/fhat_reco['mean']
        dlnfdln1pz_values = np.zeros((N,len(z_range)))
        for i in range(N):
            dlnfdln1pz_values[i] = samples_fhat_der[i]*(1+z_range)/samples_fhat[i]
        dlnf_cov = np.cov(dlnfdln1pz_values.T)
        return [Gamma, Gamma_cov, dlnf_mean, dlnf_cov]    
  
    else:
        return [Gamma, Gamma_cov]
