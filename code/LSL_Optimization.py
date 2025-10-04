# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 11:47:49 2025

This script loads project areas from a shapefile, converts geometries to
centroids for distance-based metrics, runs a multi-objective optimization
using NSGA-III to schedule project area selections over a 10-year planning
horizon, aggregates objective values across time, clusters the resulting
Pareto-efficient solutions with k-medoids, and produces a parallel coordinates
plot summarizing trade-offs between objectives.

Notes for readers:
- The optimization variables are a permutation of project areas indicating the
  order in which areas are selected. The rate vector enforces how many areas
  are selected per year.
- Objectives include overall lead risk, average spatial dispersion of selected
  areas per period (as an average pairwise distance), median child rate, and
  median income among the not-yet-selected areas. Some objectives are negated
  so the solver can treat them uniformly as minimization targets.
- After optimization, objectives are standardized and re-ordered strictly for
  analysis and plotting convenience; this does not affect the optimization.
"""

#%% Import Packages
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import math
from platypus import NSGAIII, Problem, unique, nondominated, Permutation, Archive
from platypus import CompoundOperator, PMX, Insertion, Swap
from itertools import combinations
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import random
import matplotlib.font_manager as font_manager


#%% Import project areas with attributes
project_areas = gpd.read_file('Project_Areas_with_Attributes.shp')

#%% Convert geometry of project areas
# Convert polygons to centroid points (per project area) to compute
# distance-based dispersion metrics during optimization.


# Create a copy of the project areas GeoDataFrame and convert to points
project_areas_points = project_areas.copy()

# Convert from polygons to points
project_areas_points['geometry'] = project_areas_points.geometry.centroid
# Quick diagnostic plot of project areas colored by 'Group'.
project_areas_points.plot(column='Group')

# Add the x and y coordinates of the centroid as attributes
project_areas_points['x_cord']=project_areas_points.loc[:,'geometry'].x
project_areas_points['y_cord']=project_areas_points.loc[:,'geometry'].y


#%% Set up parameters for optimization
# Define the yearly selection rate and total number of units (project areas)

even_per_year = int(8) #Number of project areas selected per year. 8 selected since 80 project areas in 10 years
rate = np.repeat(even_per_year,10) #A vector containing the number of project areas in each year. 10 selected because planning horizon is 10 years
num_units = len(project_areas_points) #Define the total number of project areas


#%% Define Functions for Optimization

# Function to determine Euclidean distance between two points
def dist(p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def objective_function(x):
    """
    Compute multi-objective values for a given permutation of project areas.

    Parameters
    ----------
    x : sequence
        Decision vector where x[0] is a permutation of indices (order of
        project area selection across the planning horizon).

    Returns
    -------
    list
        [overall_lead_risk, -integral_dist, integral_p50_chld, -integral_p50_income]
        where negatives indicate objectives that are conceptually maximized but
        are negated to fit a minimization solver.
    """

    # Get the sequence/order of project area selections (decision permutation)
    rep_all = np.array(x[0]) #rng.choice(possible, len(possible), replace=False) # 
    #rep_all = rng.choice(possible, len(possible), replace=False)

    # Get the number of project areas
    [num_units,c] = project_areas_points.shape

    # Rate of replacement: number of selections per period (year)
    # Define x[1] to be the number of years to finish doing the replacements
    inds_rep = rate

    num_periods = len(rate)#int(np.floor(len(x[0])/num_per_period)) #int(x[1])

    # Time-series aggregates for each objective across periods
    avg_distance= np.zeros((num_periods,1))
    lead_med = np.zeros((num_periods,1))
    child_med = np.zeros((num_periods,1))
    income_med = np.zeros((num_periods,1))

    data=project_areas_points['Bay_Ld_Est'].values    
    #data_adi=project_areas_points['Chld_Rt'].values

    # Loop through each period (year)
    for i in range(num_periods):        
       
        try:
            rep=rep_all[0:np.sum(inds_rep[0:i])]
        except:
            rep=rep_all
        
        # Use boolean indexing to exclude already-selected indices
        mask = np.ones(len(data), dtype=bool)
        if i>0:
            mask[rep] = False
                
        child_med[i] = np.median(project_areas_points.loc[mask,'Chld_Rt'])
        income_med[i] = np.median(project_areas_points.loc[mask,'Med_Income'])
        lead_med[i] = np.median(project_areas_points.loc[mask,'Bay_Ld_Est'])
        
        # Indices for the project areas selected in the current period
        rep_now = rep_all[np.sum(inds_rep[0:i]):np.sum(inds_rep[0:i+1])]
        
        x_cord = list(project_areas_points.iloc[rep_now,5].values.flatten())
        y_cord = list(project_areas_points.iloc[rep_now,6].values.flatten())
        
        points = list(zip(x_cord,y_cord))
        # Average pairwise distance as a measure of spatial dispersion
        distances = [dist(p1, p2) for p1, p2 in combinations(points, 2)]
        avg_distance[i] = sum(distances) / len(distances)
        
        
    integral_dist = np.sum(avg_distance) #Mean of average distance between replacements
    integral_p50_chld = np.sum(child_med)  #Integral of 90th percentile under 5
    integral_p50_income = np.sum(income_med)  #Integral of 90th percentile income
    overall_lead_risk = np.sum(lead_med)
    
    return [overall_lead_risk, -integral_dist, integral_p50_chld, -integral_p50_income ]

#%% Run Optimization
problem = Problem(1, 4)
problem.types[0] = Permutation(range(num_units))
problem.function = objective_function
problem.directions[0] = Problem.MINIMIZE
problem.directions[1] = Problem.MINIMIZE #Actually maximize but made it negative
problem.directions[2] = Problem.MINIMIZE 
problem.directions[3] = Problem.MINIMIZE #Actually maximize but made it negative


arc = Archive()

#See config line 30 to see the operators to use 0for a specific variable type
algorithm = NSGAIII(problem, divisions_outer = 12, variator=CompoundOperator(PMX(), Insertion(), Swap()))#,archive=arc)

objectives = []
variables = []

# Set a random seed for reproducibility of the evolutionary algorithm
random.seed(10)

print('optimizing')

algorithm.run(100000)
for solution in unique(nondominated(algorithm.result)):
    print(solution.variables, solution.objectives)
    objectives.append(list(solution.objectives))
    variables.append([solution.variables[0]])

print(len(objectives))

#Convert objectives into an array
objectives = np.array(objectives)
# Negate columns that were minimized to recover their conceptual maxima
objectives[:,1] = -objectives[:,1]
objectives[:,3] = -objectives[:,3]

objectives_all = objectives.copy()
variables_all = variables.copy()

#%%
# Post-processing and plotting utilities

def standardize(x):
    """Min-max scale a 1D array to [0, 1]."""
    return (x-np.min(x))/(np.max(x)-np.min(x))


objs_std = np.zeros((4,len(objectives)))
objs_std[0,:]=standardize(objectives[:,0]).flatten()
objs_std[1,:]=standardize(objectives[:,1]).flatten()
objs_std[2,:]=standardize(objectives[:,2]).flatten()
objs_std[3,:]=standardize(objectives[:,3]).flatten()

# Flip distance and income so that lower is better for all objectives,
# aligning visual interpretation across axes
objs_std[1,:]= .5 - (objs_std[1,:] - .5)
objs_std[3,:]= .5 - (objs_std[3,:] - .5)

plt.figure()
plt.plot(objs_std)
plt.xticks([0,1,2,3],['90 Percentile','Distance','U5','Income'])


#%% Only do this once!
# Reorder standardized objectives for consistent panel ordering in plots

objs_std_adjust = np.zeros_like(objs_std)

objs_std_adjust[0,:] = objs_std[0,:] #Leave lead in the same order
objs_std_adjust[1,:] = objs_std[2,:] #Make children second
objs_std_adjust[2,:] = objs_std[3,:] #Make income third
objs_std_adjust[3,:] = objs_std[1,:] #Make distance last

objs_std = objs_std_adjust


#%% Do KMedoids and get clusters of solutions

# Evaluate silhouette scores across a range of cluster counts to select k

# Create list to store silhouette scores
sil_score = [] 
for i in range(2,10):
    print(i)
    kmed = KMedoids(n_clusters=i, random_state=0).fit(objs_std.transpose())
    sil_score.append(silhouette_score(objs_std.transpose(),kmed.labels_))
sil_score = np.array(sil_score)
clust_opt = np.where(sil_score==np.max(sil_score))[0][0]+2    

plt.figure()
plt.plot(sil_score)

kmed = KMedoids(n_clusters=clust_opt, random_state=0).fit(objs_std.transpose())

labels=kmed.labels_
centers=kmed.cluster_centers_

# Find the row indices that correspond to medoids (cluster exemplars)
inds_center=[]
for i in range(clust_opt):
    inds_center.append(np.where((objs_std.transpose()[:,None]==centers[i,:]).all(axis=2))[0])

inds_center=np.array(inds_center).flatten()

#%% Make the parallel coordinates plot of clustered strategies

labels=kmed.labels_

colors=np.array([
        [191,87,0], #Orange
        #[248,151,31],
        #[255,214,0],
        [87,157,66], #Dark Green
        [166,205,87],
        [248,151,31],
        [0,169,183],
        [0,95,134],
        [51,63,72],
        [156,173,183],
        [214,210,196],
        ])/255

fig,ax =  plt.subplots(figsize=(6,4))
for i in range(clust_opt):
    ax.plot(objs_std[:,labels==i],color=colors[i,:],alpha=.06)

for i in range(clust_opt):
    ax.plot(centers.transpose()[:,i],color=colors[i,:],linewidth=3,label = 'Strategy ' + str(i+1))
ax.set_xticks([0,1,2,3],['Lead risk','Children rate','Median income','Distance'],font='Arial')
ax.tick_params(axis='x', labelsize=12)
ax.set_ylabel('Normalized objective score',font='Arial',fontsize=14)
font = font_manager.FontProperties(family='Arial',
                                   #weight='bold',
                                   style='normal', size=11)
ax.arrow(3.2,0.9,0,-0.7, head_width=0.08, head_length=0.05, color='k', lw=1.5)
ax.annotate('Direction of preference', xy=(3.4,0.5), ha='center', va='center', rotation=270, fontsize=14,font='Arial')
ax.legend(bbox_to_anchor=(.79, -0.15),ncol=2,fontsize=11,frameon=False,prop=font)
for spine in ['top','bottom','left','right']:
        ax.spines[spine].set_visible(False)

fig.savefig('fig8_parallel_plot_clusters.tiff',bbox_inches = "tight",dpi=1000)


