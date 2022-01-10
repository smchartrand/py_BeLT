"""
All things plotting related.
Most of this code created by Dr. Shawn Chartrand.
"""
import math
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import scipy as sc
from scipy.optimize import curve_fit
from scipy.special import factorial
import seaborn as sns
# Librarys needed for calculations.
from scipy.stats import binned_statistic
from matplotlib.ticker import FormatStrFormatter

plt.rcParams["font.family"] = "Times New Roman"


def IntCheck(data_max, step):
    # Integer check and determing number of bins
    if (data_max).is_integer(): 
        binning = data_max / step
    else:
        binning = (math.trunc(data_max / step)) + 1
        
    arr = np.arange(0, ((binning+1)*step), step).tolist()
        
    return binning,arr


def stream(iteration, bed_particles, model_particles, x_lim, y_lim,
                available_vertices, fp_out):
    """ Plot the complete stream from 0,0 to x_lim and y_lim. Bed particles 
    are plotted as light grey and model particles are dark blue. Allows
    for closer look at state of a subregion of the stream during simulation """
    plt.clf()
    fig = plt.figure(1)
    fig.set_size_inches(20, 6.5)
    ax = fig.add_subplot(1, 1, 1, aspect='equal')
    # NOTE: xlim and ylim modified for aspec ratio -- WIP
    ax.set_xlim((-2, x_lim))
    ax.set_ylim((0, y_lim))
    
    radius_array = np.asarray((bed_particles[:,1] / 2.0), dtype=float)
    x_center = bed_particles[:,0]
    y_center_bed = np.zeros(np.size(x_center))
    plt.rcParams['image.cmap'] = 'gray'
    ## This method of plotting circles comes from Stack Overflow questions\32444037
    ## Note that the patches won't be added to the axes, instead a collection will.
    patches = []
    for x1, y1, r in zip(x_center, y_center_bed, radius_array):
        circle = Circle((x1, y1), r)
        patches.append(circle)
    p = PatchCollection(patches, color="#BDBDBD", alpha=0.9, linewidths=(0, ))
    ax.add_collection(p)
    
    x_center_m = model_particles[:,0]
    y_center_m = model_particles[:,2]
    patches1 = []
    for x2, y2, r in zip(x_center_m, y_center_m, model_particles[:,1]/2):
        circle = Circle((x2, y2), r)
        patches1.append(circle)
    p_m = PatchCollection(patches1, cmap=matplotlib.cm.RdGy, edgecolors='black')
    p_m.set_array(model_particles[:,5])
    ax.add_collection(p_m)
    ### FOR TESTING: Plot various vertex types 
    # for xc in vertex_idx:
    #     plt.axvline(x=xc, color='b', linestyle='-')
    # for xxc in available_vertices: 
    #     plt.axvline(x=xxc, color='b', linestyle='-', linewidth=0.25)
    # for green in chosen_vertex:
    #     plt.axvline(x=green, color='g', linestyle='-')
    ### 
    plt.colorbar(p_m,orientation='horizontal',fraction=0.046, pad=0.1,label='Particle Age (iterations since last hop)')
    plt.title(f'Iteration {iteration}')

    filename = f'iter{iteration}.png'
    plots_path = fp_out + filename
    plt.savefig(plots_path, format='png',)
        
    return

def flux_info(particle_flux_list, iterations, subsample, fp_out):
    plt.clf()
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(1, 1, 1)
    bins = np.arange(-0.5, 11.5, 1) # fixed bin size
    # plt.title('Histogram of Particle Flux, I = %i iterations' % iterations, fontsize=10, style='italic')
    plt.xlabel('Downstream Crossings (particle count)', fontsize=14)
    plt.ylabel('Fraction', fontsize=14)
    ax.set_xlim((-1, max(bins)+1))
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    hist, bin_edges = np.histogram(particle_flux_list, bins=bins, density=True)

    # calculate binmiddles
    bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])
    plt.bar(bin_middles,hist,color='lightgray')
    #fig.savefig('./ScriptTest/DownstreamFluxHistogram.pdf', format='pdf', dpi=600)

    # poisson function, parameter lamb is the fit parameter
    def poisson(k, lamb):
        return (lamb**k/factorial(k)) * np.exp(-lamb)

    # fit with curve_fit
    parameters, cov_matrix = curve_fit(poisson, bin_middles, hist)
    plt.plot(bin_middles, poisson(bin_middles, *parameters), color='black', marker='o', fillstyle = 'none', markersize=4, lw=0, markeredgecolor='black', markeredgewidth=1, label='Poisson PMF Fit')

    plt.legend(loc='upper right',frameon=0)
    filename = 'CrossingsDownstreamBoundaryHist.png'
    fi_path = fp_out + filename
    fig.savefig(fi_path, format='png', dpi=600)

    #####
    # flux_list_avg = np.convolve(particle_flux_list, np.ones(subsample)/subsample, mode='valid')

    # flux_list = flux_list_avg[0::subsample]
    Time = np.arange(1,  10000 + 1, subsample)
    Flux_CS = np.cumsum(particle_flux_list[0:10000])
    
    plt.clf()
    fig = plt.figure(figsize=(8,7))
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(Time, particle_flux_list[0:10000], 'lightgray')
    # plt.title('Timeseries of Particle Crossings at Downstream Boundary')
    ax1.set_xlabel('Iteration', fontsize=14)
    ax1.set_ylabel('Particle Crossings', fontsize=14)
    ax1.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
    ax2 = ax1.twinx()

    ax2.plot(Time, Flux_CS, 'black')
    ax2.set_ylabel('Particle Crossings Cumulative Sum', color='black', rotation=270, labelpad=15, fontsize=14)
    ax2.tick_params('y', colors='black')
    
    fig.tight_layout()
    filenameCS = 'CrossingDownstreamBoundary_2YAx.png'
    fiCS_path = fp_out + filenameCS
    fig.savefig(fiCS_path, format='png', dpi=600)
        
def flux_info2(particle_flux_list, particle_age_list, n_iterations, subsample, fp_out):
    plt.clf()
    fig = plt.figure(figsize=(8,7))
    ax3 = fig.add_subplot(1, 1, 1)
    #####
    # flux_list_avg = np.convolve(particle_flux_list, np.ones(subsample)/subsample, mode='valid')
    # age_list_avg = np.convolve(particle_age_list, np.ones(subsample)/subsample, mode='valid')

    # flux_list = flux_list_avg[0::subsample]
    # age_list = age_list_avg[0::subsample]
    Time = np.arange(1,  n_iterations + 1, subsample)

    fig = plt.figure(figsize=(8,7))
    ax3 = fig.add_subplot(1,1,1)
    ax3.plot(Time[0:10000], particle_flux_list[0:10000], 'lightgray')
    # plt.title('Timeseries of Particle Crossings at Downstream Boundary')
    ax3.set_xlabel('Iteration', fontsize=14)
    ax3.set_ylabel('Particle Crossings', fontsize=14)
    ax3.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
    ax4 = ax3.twinx()

    ax4.plot(Time[0:10000], particle_age_list[0:10000], 'black')
    ax4.set_ylabel('Particle Age (# of iterations)', color='black', rotation=270, labelpad=15, fontsize=14)
    ax4.tick_params('y', colors='black')
    
    fig.tight_layout()
    filename = 'CrossingsDownstreamBoundary_Age.png'
    fi_path = fp_out + filename
    fig.savefig(fi_path, format='png', dpi=600)

        
def flux_info3(particle_flux_list, particle_age_list,particle_rage_list, n_iterations, subsample, fp_out):
    plt.clf()
    fig = plt.figure(figsize=(8,7))
    ####
    flux_list_avg = np.convolve(particle_flux_list, np.ones(subsample)/subsample, mode='valid')
    age_list_avg = np.convolve(particle_age_list, np.ones(subsample)/subsample, mode='valid')
    age_range_list_avg = np.convolve(particle_rage_list, np.ones(subsample)/subsample, mode='valid')

    flux_list = flux_list_avg[0::subsample]
    age_list = age_list_avg[0::subsample]
    age_range_list = age_range_list_avg[0::subsample]

    Time = np.arange(1,  n_iterations + 1, subsample)

    # fig = plt.figure(figsize=(9,8))
    # ax5 = fig.add_subplot(1,1,1)
    # ax5.plot(Time, flux_list, 'lightgray')
    # plt.title('Timeseries of Particle Flux at Downstream Boundary')
    # ax5.set_xlabel('Numerical Step')
    # ax5.set_ylabel('Particle Flux')
    # ax5.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
    # ax6 = ax5.twinx()

    # plt.plot(Time, age_list, linewidth='0.5', color='lightgray')
    # plt.scatter(Time, age_list, s = 15, c = age_range_list, cmap = 'RdGy')
    # ax6.set_ylabel('Mean Particle Age (# of iterations)', color='black', rotation=270, labelpad=15)
    # ax6.tick_params('y', colors='black')
    # plt.colorbar(orientation='horizontal',fraction=0.046, pad=0.2,label='Particle Age Range (max age - min age)')
    
    # fig.tight_layout()
    
    x = np.arange(1,  10000 + 1, subsample)
    y = particle_age_list[0:10000]
    dydx = particle_rage_list[0:10000] 
    flux_cumsum = np.cumsum(particle_flux_list)

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, axs = plt.subplots(figsize=(8,5))

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(dydx.min(), dydx.max())
    lc = LineCollection(segments, cmap='RdGy', norm=norm)
    # Set the values used for colormapping
    lc.set_array(dydx)
    lc.set_linewidth(1)
    line = axs.add_collection(lc)

    # plt.title('Timeseries of Particle Crossing at Downstream Boundary')
    axs.autoscale(enable=True, axis='both')
    axsTwin = axs.twinx()
    axs.set_xlabel('Iteration', fontsize=14)
    axs.set_ylabel('Mean Particle Age (# of iterations)', fontsize=14)
    axsTwin.set_ylabel('Particle Crossings Cumulative Sum', color='black', rotation=270, labelpad=15, fontsize=14)
    fig.colorbar(line, ax=axs, orientation='horizontal', pad=0.13, label='Age Range (max - min)')

    plt.plot(x[0:10000], flux_cumsum[0:10000], 'lightgray')

    # fig.tight_layout()
    

    filename = 'CrossingsDownstreamBoundary_Rage.png'
    fi_path = fp_out + filename
    fig.savefig(fi_path, format='png', dpi=600)


def heat_map(shelf, n_iterations, window_subsample, fp_out):
     # Plot heatmap of all boundary crossings
        plt.clf()
        flux_list = []
        for i in range(shelf['param']['num_subregions']):
            key = f'subregion-{i}-flux' 
            flux_full = shelf[key]
            flux_list_avg = np.convolve(flux_full, np.ones(window_subsample), mode='valid') / window_subsample
            flux_list_ss = flux_list_avg[0::window_subsample]

            flux_list.append(flux_list_ss)

        # the content of labels of these yticks
        # print(flux_list)
        flux_list = np.transpose(flux_list)
        flux_heatmap = sns.heatmap(flux_list, cmap="coolwarm")
        fig = flux_heatmap.get_figure()

        
        filename = 'FluxAllBoundaries_Heatmap.png'
        fi_path = fp_out + filename
        fig.savefig(fi_path, format='png', dpi=600)


def plot_cmsm_comp(flux_list_list, n_iterations, subsample, fp_out):

    colours = ['green', 'orange', 'red', 'blue', 'purple']
    fig, axs = plt.subplots(figsize=(9,5))
    key = 0
    for flux in flux_list_list:
  
        flux_cumsum = np.cumsum(flux)
        print(flux_cumsum)

        iteration_subsampled = np.arange(500000, 502000)

        plt.plot(iteration_subsampled, flux_cumsum[500000:502000], colours[key])
        key += 1
    
    filename = 'ComparedCumSum_Subsampled.png'
    fi_path = fp_out + filename
    fig.savefig(fi_path, format='png', dpi=600)


def binned_age_vs_elevations_metrics(simulation_id, sim_plot_loc, avg_age_path, age_range_path, full_plux_path, elevation_metric_path):
    # Written by Shawn. Tranferred from BinningAvg_SCipt.ipynb
    # Load up the light table data and give the loaded data column headers
    file_path = (avg_age_path)
    data = pd.read_table(file_path, sep='\t',header=0)
    data.replace('  ', np.nan, inplace=True)
    data.dropna(inplace=True)
    # Round the data in each column to the nearest integer value
    data.reset_index(drop=True,inplace=True)
    data_copy = data.astype('float64')
    df_shape = data.shape
    ###################################
    file_path2 = (age_range_path)
    data2 = pd.read_csv(file_path2, sep='\t',header=0)
    data2.replace('  ', np.nan, inplace=True)
    data2.dropna(inplace=True)
    # Round the data in each column to the nearest integer value
    data2.reset_index(drop=True,inplace=True)
    data2_copy = data2.astype('float64')
    df2_shape = data2.shape
    ###################################
    file_path3 = (full_plux_path)
    data3 = pd.read_csv(file_path3, sep='\t',header=0)
    data3.replace('  ', np.nan, inplace=True)
    data3.dropna(inplace=True)
    # Round the data in each column to the nearest integer value
    data3.reset_index(drop=True,inplace=True)
    data3_copy = data3.astype('float64')
    df3_shape = data3.shape
    ###################################
    file_path4 = (elevation_metric_path)
    data4 = pd.read_csv(file_path4, sep='\t',header=0)
    data4.replace('  ', np.nan, inplace=True)
    data4.dropna(inplace=True)
    # Round the data in each column to the nearest integer value
    data4.reset_index(drop=True,inplace=True)
    data4_copy = data4.astype('float64')
    df4_shape = data4.shape

    # Calc the binned averages
    # sort the arrays
    data_sort = np.sort(data, axis=1)
    data2_sort = np.sort(data2, axis=1)
    data4_sort = np.sort(data4, axis=1)
    new_data = np.ravel(data_sort)
    new_data2 = np.ravel(data2_sort)
    new_data4 = np.ravel(data4_sort)

    step = 5
    step2 = 100
    data_max = np.max(data_sort)
    data2_max = np.max(data2_sort)

    value,value1 = IntCheck(data_max, step)
    value2,value3 = IntCheck(data2_max, step2)

    # Use scipy library for averaging within bins of different variable
    bin_elev, edges, _ = binned_statistic(new_data, new_data4, 'mean', bins=value1)
    revbin = edges[:-1]

    bin_elev2, edges2, _ = binned_statistic(new_data2, new_data4, 'mean', bins=value3)
    revbin2 = edges2[:-1]

    # Define bin centers for plotting
    bin_ctr = np.arange(step / 2, ((value)*step), step).tolist()
    bin2_ctr = np.arange(step2 / 2, ((value2)*step2), step2).tolist()

    # Begin plotting
    fig = plt.figure(figsize=(8,8))
    fig.suptitle(simulation_id, fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.90)
    #fig.subplots_adjust(hspace=.4)
    fig.subplots_adjust(wspace=.4)

    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,3)
    ax3 = fig.add_subplot(2,2,2)
    ax4 = fig.add_subplot(2,2,4)

    ax1.scatter(data, data4, s=7.5, facecolors='gainsboro', edgecolors='gainsboro')
    #ax1.plot(data, data4, marker = 'o', markersize = 4, markeredgewidth=1,markeredgecolor=‘gainsboro’,markerfacecolor='None’)
    ax1.plot(bin_ctr, bin_elev, marker = 'o', markersize = 5, color='orangered', lw=0)

    ax2.plot(data2, data4, marker = 'd', markersize = 4,color='darkgray',lw=0)
    ax2.plot(bin2_ctr, bin_elev2[0:16], marker = 'o', markersize = 5, color='orangered', lw=0)
    ax3.plot(data3, data2, marker = 'o', markersize = 4, color='dodgerblue', lw=0)
    ax4.plot(data3, data4, marker = 'd', markersize = 4,color='steelblue',lw=0)

    ax1.set_xlabel('Average Particle Age (# of iterations)', fontsize=12)
    ax1.set_ylabel('Elevation Metric (mm)', fontsize=12)
    #ax1.set_xticklabels([], visible=True)
    ax1.tick_params(direction='out', length=6, width=1)
    #ax1.set_ylim(0,30)
    # ax1.set_xticks([0,40,80,120,160,200], minor=True)
    # ax1.set_yticks((0,40,80,120,160),minor=True)

    ax2.set_xlabel('Age Range (max - min))', fontsize=12)
    ax2.set_ylabel('Elevation Metric (mm)', fontsize=12)
    # ax2.set_xticklabels([], visible=True)
    ax2.tick_params(direction='out', length=6, width=1)

    ax3.set_xlabel('Total Particle Crossings in Domain (# of particles)', fontsize=12)
    ax3.set_ylabel('Age Range (max - min))', fontsize=12)
    # ax3.set_xticklabels([], visible=True)
    ax3.tick_params(direction='out', length=6, width=1)

    ax4.set_xlabel('Total Particle Crossings in Domain (# of particles)', fontsize=12)
    ax4.set_ylabel('Elevation Metric (mm)', fontsize=12)
    # ax4.set_xticklabels([], visible=True)
    ax4.tick_params(direction='out', length=6, width=1)

    fig.tight_layout(pad=1.0)
    fig.savefig(f'{sim_plot_loc}/Age_vs_Elevation_{simulation_id}.png', format='png', dpi=600)
