#Import pacakges
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Impor saves results for convergense and sensitivity analysis
import saved_results
 
# Define parameters
class params:
    pass
params.dz = 1#Cell size [m]
params.kp = 0.05 # self shading []m2/mmol N
params.kw = 0.2 #0.0375 #Specific light attenuation [1/m]
params.I_0 = 350 #Light intensity at the surface [Einstein-unit] or [umol photons/m2/s]
params.d=100 #Depth of the water column [m]
params.z=np.arange(0.5*params.dz,params.dz+params.d-0.5*params.dz,params.dz) #Make grid, start, stop, step
params.n=int(params.d/params.dz) #100 #Number of cells in grid
params.u=0.96 # 0.96 or 3? Vertical velocity [1/day]
params.D = 5 #5*60*60*24/(100**2) # Diffusion [m2/day]
params.gmax= 1.5 # maximum growth rate [1/day]
params.H_I = 20 #30 # light intensity half saturation [Einstein-unit] or [umol photons/m2/s]
params.l= 0.01*24 #loss rate / mortality [1/days] 
params.H_N=0.3#0.0425 #Nutrients half saturation [mmol nutrient/m3]
params.e = 0.5 # dimensionless Nutrient recycling coefficient
params.N_bottom = 30 #mmol N/m3
params.gamma =1.5 # grazing mortality [m3/mmol N/day]
params.phi = 75 # Lattitude, Equator = 0, Greenland sea  =75
time = [0,365] # Define time span, 2 years untill nutrients converged

######################

# Create initial vectors for P and N
P=np.zeros(params.n) # mmol N/m3
P[0]=0.5 # mmol N/m3
N=np.ones(params.n)*30 # mmol N/m3

###########################


# Function for light

# Note: to add seasonality add a 't' in the functions for light(t,P,paramas) and growth(t,P,N,params) 
# and uncomment I_season and the other I in the light function below. 

def light(P,params):
    #I_season = params.I_0 * (1-0.8*np.sin(np.pi*params.phi/180)*np.cos(2*np.pi*t/365))
    Q = params.kp * params.dz * (np.cumsum(P) + (P/2))
    I = params.I_0 * np.exp(-params.kw * params.z - Q)
    #I = I_season * np.exp(-params.kw * params.z - Q)
    return I

######################


# Function for growth
def growth(P,N,params):
    g = params.gmax*np.minimum(light(P,params)/(light(P,params)+params.H_I),N/(N+params.H_N))
    return g
 
######################

 
def fluxP(t,P,N,params):
    # Creating vectors
    Ja = np.zeros(params.n+1)
    Jd = np.zeros(params.n+1)
    J = np.zeros(params.n+1)
    dpdt=np.zeros(params.n)
    
    for i in range(params.n+1):
        if i == 0 or i == params.n:
            # Boundary conditions for advective flux
            Ja[0] = 0 
            Ja[params.n] = 0
            # Boundary conditions for diffusive flux
            Jd[0] = 0
            Jd[params.n] = 0
        else:
            Ja[i] = params.u*P[i-1]
            Jd[i] = -params.D *(P[i]-P[i-1])/params.dz
        # Combine to one J-flux
        J[i] = Ja[i] + Jd[i]
    for i in range(params.n):   
        dpdt[i] = -(J[i+1] - J[i])/params.dz
    dpdt = (dpdt) + growth(P,N,params)*P-params.l*P-params.gamma*P**2
    return dpdt
 

######################

 

def fluxN(t,P,N,params):
    J = np.zeros(params.n+1)
    dNdt=np.zeros(params.n)  
    for i in range(params.n+1):
        if i == 0 or i == params.n:
            # Boundary conditions for the diffusive flux
            J[0] = 0
            J[params.n] = -params.D*((params.N_bottom-N[params.n-1])/params.dz)
        else:
            J[i] = -params.D *(N[i]-N[i-1])/params.dz
    for i in range(params.n):   
        dNdt[i] = -(J[i+1] - J[i])/params.dz
    dNdt = (dNdt)-growth(P,N,params)*P+params.l*P*params.e
    return dNdt


##################### 

 
y=np.concatenate((P,N))

 
def fluxtot(t,y,params):
    P = y[0:params.n]
    N = y[params.n:2*params.n]
    dydt=np.concatenate([fluxP(t,P,N,params),fluxN(t,P,N,params)])
    return dydt

 
#####################

result=solve_ivp(fluxtot,time,y,args=(params,))

# Define functions for plotting
I=light(P,params)
fI=I/(I+params.H_I)
fN=result['y'][params.n:params.n*2,-1]/(result['y'][params.n:params.n*2,-1]+params.H_N)


#%% Plotting

# Phytoplankton surface plot 
plt.figure(dpi=1200)
plt.pcolor(result['t'],-params.z,result['y'][0:params.n,:])
plt.colorbar()
plt.title('Phytoplankton [mmol N/m3]')
plt.xlabel('Time [days]')
plt.ylabel('Depth [m]')
plt.show()

# Nutrients surface plot
plt.figure(dpi=1200)
plt.pcolor(result['t'],-params.z,result['y'][params.n:params.n*2,:])
plt.colorbar()
plt.title('Nutrients [mmol N/m3]')
plt.xlabel('Time [days]')
plt.ylabel('Depth [m]')
plt.show()

# Light
plt.figure(dpi=1200)
plt.plot(I,-params.z,'black')
plt.xlabel('Light [Einstein unit]')
plt.ylabel('Depth [m]')
plt.title('Light')
plt.show()

# Phytoplankton
plt.figure(dpi=1200)
plt.plot(result['y'][0:params.n,-1],-params.z,'black')
plt.xlabel('Phytoplankton [mmol N/m3]')
plt.ylabel('Depth [m]')
plt.title('Phytoplankton converged')
plt.show()

# Nutrients
plt.figure(dpi=1200)
plt.plot(result['y'][params.n:params.n*2,-1],-params.z,'black')
plt.xlabel('Nutrients [mmol N/m3]')
plt.ylabel('Depth [m]')
plt.title('Nutrients converged')
plt.show()

# # Light in the absence of shading
plt.figure(dpi=1200)
plt.plot(saved_results.I_noshade,-params.z,'black',linestyle='--',label='Light in the absence of shading')
plt.plot(I,-params.z,'black',label='Light')
plt.xlabel('Light [Einstein unit]')
plt.ylabel('Depth [m]')
plt.ylim([-10,0])
plt.legend()
plt.title('Light in the absence of shading')
plt.show()

# Functional response
plt.figure(dpi=1200)
plt.plot(fI,-params.z,'black',linestyle='--',label='I/(I+H)')
plt.plot(fN,-params.z,'black',label='N/(N+H)')
plt.xlabel('Functional response')
plt.ylabel('Depth [m]')
plt.title('Functional response')
plt.axhline(-10.5,color='black',linewidth=0.5)
plt.legend()
plt.show()

# Convergence Analysis
plt.figure(dpi=1200)
plt.plot(saved_results.P_4,-saved_results.Z_4,label='$\Delta z$ = 4.0 m',linewidth=1.0,color='black')
plt.plot(saved_results.P_2,-saved_results.Z_2,label='$\Delta z$ = 2.0 m',linewidth=0.8,color='black')
plt.plot(saved_results.P_1,-saved_results.Z_1,label='$\Delta z$ = 1.0 m',linewidth=0.6,color='black')
plt.plot(saved_results.P_05,-saved_results.Z_05,label='$\Delta z$ = 0.5 m',linewidth=0.4,color='black')
plt.plot(saved_results.P_025,-saved_results.Z_025,label='$\Delta z$ = 0.25 m',linewidth=0.2,color='black')
plt.xlabel('Phytoplankton [mmol N/m3]')
plt.ylabel('Depth [m]')
plt.title('Convergence analysis - grid size')
plt.ylim([-16,-8])
plt.xlim([0.125, 0.275])
plt.legend()
plt.show()


# Sensitivity analysis
plt.figure(dpi=1200)
plt.plot(result['y'][0:params.n,-1],-params.z,label='Baseline',linewidth=0.1,color='black')
plt.plot(saved_results.gmax10,-params.z,label='increased 10%',linewidth=0.2,color='black')
plt.plot(saved_results.gmax20,-params.z,label='increased 20%',linewidth=0.3,color='black')
plt.plot(saved_results.gmax30,-params.z,label='increased 30%',linewidth=0.4,color='black')
plt.plot(saved_results.gmax40,-params.z,label='increased 40%',linewidth=0.5,color='black')
plt.plot(saved_results.gmax50,-params.z,label='increased 50%',linewidth=0.6,color='black')
plt.plot(saved_results.gmax60,-params.z,label='increased 60%',linewidth=0.7,color='black')
plt.plot(saved_results.gmax70,-params.z,label='increased 70%',linewidth=0.8,color='black')
plt.plot(saved_results.gmax80,-params.z,label='increased 80%',linewidth=0.9,color='black')
plt.plot(saved_results.gmax90,-params.z,label='increased 90%',linewidth=1.0,color='black')
plt.plot(saved_results.gmax100,-params.z,label='increased 100%',linewidth=1.1,color='black')
plt.xlabel('Phytoplankton [mmol N/m3]')
plt.ylabel('Depth [m]')
plt.title('Sensitivity analysis - increasing gmax')
plt.ylim([-30,0])
plt.legend()
plt.show()

