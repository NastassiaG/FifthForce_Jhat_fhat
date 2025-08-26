from scipy import optimize, stats, interpolate
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic #WhiteKernel, ConstantKernel

from scipy.interpolate import splev, splrep

from iminuit import Minuit


# chi_squared_spline(...) computes the chi squared for a spline defined through values farray at knots zarray, given data points fhat_data at redshifts z_data with covariance fhat_data_covariance
def chi_squared_spline(farray, zarray = np.array([0.295, 0.467, 0.626, 0.771]), k = 3, 
    z_data = np.array([0.02, 0.025, 0.067, 0.10, 0.15, 0.32, 0.38, 0.44, 0.57, 0.59, 0.70, 0.73, 0.74, 0.76, 0.85, 0.978, 1.05, 1.40, 1.48, 1.944]), 
    fhat_data = np.array([0.398, 0.39, 0.423, 0.37, 0.53, 0.384, 0.497, 0.413, 0.453, 0.488, 0.473, 0.437, 0.50, 0.440, 0.52, 0.379, 0.280, 0.482, 0.30, 0.364]), 
    fhat_data_errors = np.array([0.065, 0.11, 0.055, 0.13, 0.16, 0.095, 0.045, 0.080, 0.022, 0.060, 0.041, 0.072, 0.11, 0.040, 0.10, 0.176, 0.080, 0.116, 0.13, 0.106]), fhat_data_covariance = [], use_covariance = False):
    
    # create diagonal covariance matrix if errors are given instead of a covariance
    if use_covariance == False:
        fhat_data_covariance = np.diag(fhat_data_errors**2)
        
    if len(zarray) == 3:
        # Create a cubic spline interpolation object
        spline = interpolate.CubicSpline(zarray, farray)
        fhat_spline = spline(z_data)
    else:
        spl = splrep(zarray, farray, k=k)
        fhat_spline = splev(z_data, spl)
        
    chi_squared = (fhat_spline - fhat_data).T @ np.linalg.inv(fhat_data_covariance) @ (fhat_spline - fhat_data)
    
    return chi_squared


# minuit_spline(...) computes the best fit values of the function at knots zarray, given data points fhat_data at redshifts z_data with covariance fhat_data_covariance
def minuit_spline(zarray = np.array([0.295, 0.467, 0.626, 0.771]), k = 3, 
    z_data = np.array([0.02, 0.025, 0.067, 0.10, 0.15, 0.32, 0.38, 0.44, 0.57, 0.59, 0.70, 0.73, 0.74, 0.76, 0.85, 0.978, 1.05, 1.40, 1.48, 1.944]), 
    fhat_data = np.array([0.398, 0.39, 0.423, 0.37, 0.53, 0.384, 0.497, 0.413, 0.453, 0.488, 0.473, 0.437, 0.50, 0.440, 0.52, 0.379, 0.280, 0.482, 0.30, 0.364]), 
    fhat_data_errors = np.array([0.065, 0.11, 0.055, 0.13, 0.16, 0.095, 0.045, 0.080, 0.022, 0.060, 0.041, 0.072, 0.11, 0.040, 0.10, 0.176, 0.080, 0.116, 0.13, 0.106]), fhat_data_covariance = [], use_covariance = False):

    
    def chi_squared(*farray):
        return chi_squared_spline(farray, zarray = zarray, k=k, z_data = z_data, 
                                   fhat_data = fhat_data, fhat_data_errors=fhat_data_errors, fhat_data_covariance = fhat_data_covariance, use_covariance = use_covariance)
    
    farray = np.full(len(zarray),1)

    #Create a Minuit object with chi-squared function
    minuit = Minuit(chi_squared, *farray)

    # Perform the minimization
    minuit.migrad()

    # Get the optimized parameters
    best_fit_values = minuit.values

    # Get the covariance matrix
    covariance_matrix = minuit.covariance
    
    return [best_fit_values, covariance_matrix]

# spline_reconstruction(...) draws N samples from a multivariate Gaussian defined by best_fit_values and covariance_matrix; for each sample, a spline interpolation is performed; then, the best fit values and the covariance of the function values (and the derivative values) at z_range are obtained from the 1000 samples

def spline_reconstruction(best_fit_values, covariance_matrix, zarray = np.array([0.295, 0.467, 0.626, 0.771]), z_range = np.linspace(0,2,100), N = 10000, return_covariance = False):
    # Generate N different samples
    samples = np.random.multivariate_normal(best_fit_values, covariance_matrix, size = N)
        
    #spline interpolation for each of these samples
    spline = np.zeros((N,len(z_range)))
    spline_der = np.zeros((N,len(z_range)))
    
    if len(zarray) == 3:
        for i in range(N):
            spl = interpolate.CubicSpline(zarray, samples[i])
            spline[i] = spl(z_range)
            spline_der[i] = spl.derivative()(z_range)
        
    else:
        for i in range(N):
            spl = splrep(zarray, samples[i])
            spline[i]=splev(z_range, spl)
            spline_der[i] = splev(z_range, spl, der=1)
    
    # Means of interpolated values along whole redshift range
    means=[np.mean(spline[:,i]) for i in range(len(z_range))] 
    means_der=[np.mean(spline_der[:,i]) for i in range(len(z_range))]
    
    if return_covariance == False:
        stdev=[np.std(spline[:,i]) for i in range(len(z_range))]   
        stdev_der=[np.std(spline_der[:,i]) for i in range(len(z_range))]
        result = {
            'x' : z_range,
            'mean': np.array(means),
            'stdev': np.array(stdev),
            'mean_d1': np.array(means_der),
            'stdev_d1': np.array(stdev_der)
        }
        
    else:
        spline_01 = np.block([spline, spline_der])
        tot_cov = np.cov(spline_01.T)
        covariance_matrix = tot_cov[0:len(z_range),0:len(z_range)]
        covariance_matrix_der = tot_cov[len(z_range):,len(z_range):]
        covariance_matrix_01 = tot_cov[0:len(z_range),len(z_range):] #cross covariance f and f'
        
        result = {
            'x' : z_range,
            'mean': np.array(means),
            'cov': np.array(covariance_matrix),
            'mean_d1': np.array(means_der),
            'cov_d1': np.array(covariance_matrix_der),
            'cov_01': np.array(covariance_matrix_01)
        }
    
    return result
    


