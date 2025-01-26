"""

partial reproduction of litwin-kumar et al. doiron 2016

the point isnt to reproduce their paper, but to reproduce ssa_fs.py
"""

from __future__ import division

import matplotlib.gridspec as gridspec

import numpy as np
import matplotlib.pyplot as plt
# import scipy as sp

import argparse
# import brian2 as b2
from brian2 import *


def get_spikes(start_time, end_time, sim_dt, spikemon_t):
    """
    get firing rate given spikemonitor and time interval.

    start_time: time of input start
    end_time: time of input end
    sim_dt: simulation time step
    spikemon_t: time array from spike monitor
    """

    start_idx = int(start_time / sim_dt)
    end_idx = int(end_time / sim_dt)

    # convert spikemon to ms time units
    ms_units = (spikemon_t / ms)

    # mark all times where there was a spike between the start and end times. add everything.
    total_spikes = np.sum((ms_units > start_time) * (ms_units < end_time))

    return total_spikes


def get_FR(start_time, end_time, sim_dt, spikemon_t):
    """
    get firing rate given spikemonitor and time interval.

    start_time: time of input start
    end_time: time of input end
    sim_dt: simulation time step
    spikemon_t: time array from spike monitor
    """

    start_idx = int(start_time / sim_dt)
    end_idx = int(end_time / sim_dt)

    # convert spikemon to ms time units
    ms_units = (spikemon_t / ms)

    # mark all times where there was a spike between the start and end times. add everything.
    total_spikes = np.sum((ms_units > start_time) * (ms_units < end_time))

    # convert ms time interval to sec
    time_interval = (end_idx - start_idx) / 1000.

    # convert to spikes per time unit
    firing_rate = total_spikes * sim_dt / ms

    return firing_rate


def collect_spikes(stim_array, stim_dt, sim_dt, spikemon_t):
    """
    get total spike count for a given spikemon

    stim_array: array of stimulation time/strengths, e.g., [0,0,0,0,1,0,0,0,1,0,0]
    stim_dt: time interval of stimulation
    sim_dt: time interval of simulation
    spikemon_t: time array from spike monitor

    """

    # get all stim start times (index position*stim_dt)
    stim_start_times = np.where(stim_array != 0)[0] * stim_dt

    # preallocate firing rate array
    spike_array = np.zeros(len(stim_start_times))

    for i in range(len(stim_start_times)):
        spike_array[i] = get_spikes(stim_start_times[i], stim_start_times[i] + stim_dt, sim_dt, spikemon_t)

    return spike_array


def collect_FR(stim_array, stim_dt, sim_dt, spikemon_t):
    """
    get all firing rates for a given spikemon

    stim_array: array of stimulation time/strengths, e.g., [0,0,0,0,1,0,0,0,1,0,0]
    stim_dt: time interval of stimulation
    sim_dt: time interval of simulation
    spikemon_t: time array from spike monitor

    """

    # get all stim start times (index position*stim_dt)
    stim_start_times = np.where(stim_array != 0)[0] * stim_dt

    # preallocate firing rate array
    FR_array = np.zeros(len(stim_start_times))

    for i in range(len(stim_start_times)):
        FR_array[i] = get_FR(stim_start_times[i], stim_start_times[i] + stim_dt, sim_dt, spikemon_t)

    return FR_array


"""
def mean_FR(start_time,dur,dt,spikemon_t):

    for i in range(len(stim_start_times)):
        FR_array[i] = get_FR(stim_start_times[i],stim_start_times[i]+stim_dt,sim_dt,spikemon_t)

    return FR_array
"""


def setup_and_run(pv_opto1=0, som_opto1=0, single_tone=False, q=.4, multiplier=2):
    # start_scope()
    np.random.seed(2)

    # multiplier = 1

    n_pyr = 400 * multiplier
    n_pv = 100 * multiplier
    n_som = 100 * multiplier

    vE = 0. * mV
    vI = -67 * mV

    # weight matrix
    multiplier = 20.  # 20.
    W = np.zeros((3, 3))
    W[0, 0] = 1  # .5#0.9 # wee
    W[0, 1] = 2  # 2.#2. # wep
    W[0, 2] = 1  # .5#0.5 # wes

    W[1, 0] = .1  # .1#.1 # wpe
    W[1, 1] = 2  # 1.#2. # wpp
    W[1, 2] = 2  # .6#0.6 # wps

    W[2, 0] = 6  # 10.#8 # wse
    W[2, 1] = 0.
    W[2, 2] = 0.

    W *= multiplier

    b = 8 * pA

    # a = 4.*nS
    tauw = 150. * ms
    gsd = 18.75 * nS

    gee_max = W[0, 0] * nS  # 1.66*nS
    gep_max = W[0, 1] * nS  # 136.4*nS
    ges_max = W[0, 2] * nS  # 68.2*nS

    gpe_max = W[1, 0] * nS  # 5.*nS
    gpp_max = W[1, 1] * nS  # (136.4*nS)/10
    gps_max = W[1, 2] * nS  # 45.5*nS

    gse_max = W[2, 0] * nS  # 5*(1.66*nS)
    # gsp_max = 136.4*nS
    # gss_max = 45.5*nS

    EL = -60 * mV

    sigma = 20 * mV  # 20*mV

    tauD_fast = 10 * ms  # 11*ms#11*ms

    tauD_slow1 = 1000 * ms  # 1500*ms#800*ms
    tauD_slow2 = 250 * ms  # 200*ms#200*ms#150*ms

    tauI_fast = 1 * ms  # 5*ms

    # timed inputs
    # q = .#.35
    # 50-60...90-100,  120-130...160-170

    if not (single_tone):
        stim_arr = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    else:
        stim_arr = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    stim_dt = 10

    stimulus = TimedArray(
        stim_arr,
        dt=stim_dt * ms)

    stim_arr[np.where(stim_arr != 0)[0] + 1] = 1
    stimulusb = TimedArray(
        stim_arr,
        dt=stim_dt * ms)

    pv_opto1 = pv_opto1 * nA
    som_opto1 = som_opto1 * nA

    print('pv_opto =', pv_opto1, '; som_opto =', som_opto1)

    eqs_e = '''
    # E soma
    dv/dt=( -w + Itot + gL*dT*exp((v-vT)/dT) + sigma*xi*(nS*Cm)**.5 )/Cm : volt (unless refractory)

    iL = -gL*(v-EL) : amp
    iDend = -gsd*(v-vD)/(0.3) : amp

    # total current
    Itot = iDend + iSynEE + iSynEP + iL + I + q*IFast*nA*DSlow*DFast: amp

    # soma synapses
    iSynEE = -gee*(v-vE) : amp
    iSynEP = -gep*(v-vI) : amp

    # synapse decay
    dgee/dt = -gee/ms : siemens
    dgep/dt = -gep/ms : siemens
    dges/dt = -ges/ms : siemens

    # adaptation
    dw/dt = (a*(v-EL) - w)/tauw : amp

    # dendrite
    dvD/dt = (iDendL + iSoma + iDendSyn)/Cm : volt
    iDendL = -gL*(vD-EL) : amp
    iSoma = -gsd*(vD-v)/(0.7) : amp
    iDendSyn = -ges*(vD-vI) : amp

    # synaptic depression (thalamus, slow)
    dDSlow/dt = (1-DSlow)/tauD_slow1 - DSlow*stimulus(t)/tauD_slow2 : 1

    # synaptic depression (thalamus, fast)
    dDFast/dt = (1-DFast - stimulusb(t))/tauD_fast : 1

    # fast input facilitation
    dIFast/dt = (-IFast + stimulus(t))/tauI_fast : 1

    gL : siemens
    vT : volt
    dT : volt
    a : siemens
    I : amp
    Cm : farad
    '''

    eqs_p = '''
    # PV
    dv/dt=( q*IFast*nA*DSlow*DFast + I - pv_opto1 - w + iL + gL*dT*exp((v-vT)/dT) + iSynPE + iSynPP + iSynPS + sigma*xi*(nS*Cm)**.5)/Cm : volt (unless refractory)
    iL = -gL*(v-EL) : amp

    # soma synapses
    iSynPE = -gpe*(v-vE) : amp
    iSynPP = -gpp*(v-vI) : amp
    iSynPS = -gps*(v-vI) : amp

    # synapse decay
    dgpe/dt = -gpe/(25*ms) : siemens
    dgpp/dt = -gpp/ms : siemens
    dgps/dt = -gps/ms : siemens

    # adaptation
    dw/dt = (a*(v-EL) - w)/tauw : amp

    # synaptic depression (thalamus, slow)
    dDSlow/dt = (1-DSlow)/tauD_slow1 - DSlow*stimulus(t)/tauD_slow2 : 1

    # synaptic depression (thalamus, fast)
    dDFast/dt = (1-DFast - stimulus(t))/tauD_fast : 1

    # fast input facilitation
    dIFast/dt = (-IFast + stimulus(t))/tauI_fast : 1

    pv_opto : amp
    gL : siemens
    vT : volt
    dT : volt
    a : siemens
    I : amp
    Cm : farad
    '''

    eqs_s = '''
    # SOM
    dv/dt=( gL*dT*exp((v-vT)/dT) + I - som_opto1 - w + iL + iSynSE + sigma*xi*(nS*Cm)**.5)/Cm : volt (unless refractory)
    iL = -gL*(v-EL) : amp

    # soma synapses
    iSynSE = -gse*(v-vE) : amp
    #iSynPP = -gpp*(v-vI) : amp
    #iSynPS = -gps*(v-vI) : amp

    # synapse decay
    dgse/dt = -gse/(15*ms) : siemens # dgse/dt = -gse/(15*ms) : siemens
    #dgpp/dt = -gpp/ms : siemens
    #dgps/dt = -gps/ms : siemens

    # adaptation
    dw/dt = (a*(v-EL) - w)/tauw : amp

    som_opto : amp
    gL : siemens
    vT : volt
    dT : volt
    a : siemens
    I : amp
    Cm : farad
    '''

    ############################### Neuron groups and parameters
    G_PYR = NeuronGroup(n_pyr, eqs_e, threshold='v>20*mV', reset='v=-60*mV;w+=b', method='Euler',
                        refractory=2 * ms)  # pyr
    # pyr refractory period ~8-9ms (The Refractory Periods and Threshold Potentials... Chen, Chen, Wu, Wang, 2005)
    # pyr refractory period ~5-6ms (Synaptic Refractory Period Provides... Hjelmstad, Nicoll, Malenka, 1997)

    G_PYR.Cm = 180 * pF
    G_PYR.gL = 6.25 * nS
    G_PYR.dT = 1. * mV  # 0.25 for PV
    G_PYR.vT = -40. * mV  # randomly distributed and different for SOMs
    G_PYR.a = 4. * nS

    G_PV = NeuronGroup(n_pv, eqs_p, threshold='v>20*mV', reset='v=-60*mV', method='Euler', refractory=2 * ms)
    G_PV.Cm = 80. * pF
    G_PV.gL = 5. * nS
    G_PV.dT = 0.25 * mV
    G_PV.vT = -40. * mV
    G_PV.a = 0. * nS
    # G_PV.pv_opto = pv_opto1*nA

    G_SOM = NeuronGroup(n_som, eqs_s, threshold='v>20*mV', reset='v=-60*mV', method='Euler', refractory=2 * ms)
    G_SOM.Cm = 80. * pF
    G_SOM.gL = 5. * nS
    G_SOM.dT = 1. * mV
    G_SOM.vT = -45. * mV
    G_SOM.a = 4. * nS
    # G_SOM.som_opto = som_opto1*nA

    ############################### inputs
    # baseline
    G_PYR.I[:] = .35 * nA  # .35*nA
    G_PV.I[:] = .05 * nA  # .1*nA#.05*nA
    G_SOM.I[:] = .1 * nA  # .025*nA#.025*nA

    ############################### synapses
    # inhibitory conductances are incremented positively here, but they give rise to negative currents based on the inhibitory reversal potential in the equations above.
    SynEE = Synapses(G_PYR, G_PYR, on_pre='gee += gee_max/n_pyr')
    SynEE.connect(p=.1)
    SynEP = Synapses(G_PV, G_PYR, on_pre='gep += gep_max/n_pv')
    SynEP.connect(p=.6)
    SynES = Synapses(G_SOM, G_PYR, on_pre='ges += ges_max/n_som')
    SynES.connect(p=.6)

    SynPE = Synapses(G_PYR, G_PV, on_pre='gpe += gpe_max/n_pyr')
    SynPE.connect(p=.6)
    SynPP = Synapses(G_PV, G_PV, on_pre='gpp += gpp_max/n_pv')
    # SynPP = Synapses(G_PV,G_PV,on_pre='gpp -= 0*nS')
    SynPP.connect(p=.6)
    SynPS = Synapses(G_SOM, G_PV, on_pre='gps += gps_max/n_som')
    # SynPS = Synapses(G_SOM,G_PV,on_pre='gps -= 0*nS')
    SynPS.connect(p=.6)

    SynSE = Synapses(G_PYR, G_SOM, on_pre='gse += gse_max/n_pyr')
    SynSE.connect(p=.6)
    # SynSP = Synapses(G_PYR,G_PV,on_pre='gep -= gep_max')
    # SynSP.connect(p=.6)
    # SynSS = Synapses(G_PYR,G_SOM,on_pre='gep -= gep_max')
    # SynSS.connect(p=.6)

    ############################### run
    G_PYR.v[:] = EL + 10 * mV
    G_PYR.vD[:] = EL + 10 * mV
    G_PYR.DSlow[:] = 1
    G_PYR.DFast[:] = 1

    G_PV.DSlow[:] = 1
    G_PV.DFast[:] = 1

    G_PV.v[:] = EL
    G_SOM.v[:] = EL
    # q*IFast*nA*DSlow*DFast
    M_PYR = StateMonitor(G_PYR, ['v', 'vD', 'w', 'ges',
                                 'DSlow', 'DFast', 'IFast',
                                 'iSynEE', 'iSynEP'], record=True)
    M_PV = StateMonitor(G_PV, ['v', 'DSlow', 'DFast'], record=True)
    M_SOM = StateMonitor(G_SOM, ['v'], record=True)

    spikemon_PYR = SpikeMonitor(G_PYR)
    spikemon_PV = SpikeMonitor(G_PV)
    spikemon_SOM = SpikeMonitor(G_SOM)

    defaultclock.dt = 0.1 * ms
    T = len(stim_arr) * stim_dt * ms
    run(T)

    fulldict = {}

    fulldict['q'] = q
    fulldict['som_opto'] = som_opto1
    fulldict['pv_opto'] = pv_opto1

    fulldict['M_PYR'] = M_PYR
    fulldict['M_PV'] = M_PV
    fulldict['M_SOM'] = M_SOM

    fulldict['spikemon_PYR'] = spikemon_PYR
    fulldict['spikemon_PV'] = spikemon_PV
    fulldict['spikemon_SOM'] = spikemon_SOM

    fulldict['stim_arr'] = stim_arr
    fulldict['stim_dt'] = stim_dt

    fulldict['T'] = T
    fulldict['defaultclock'] = defaultclock.dt

    return fulldict


def plot(fulldict, choice):
    # fulldict = setup_and_run()

    M_PYR = fulldict['M_PYR']
    M_PV = fulldict['M_PV']
    M_SOM = fulldict['M_SOM']

    spikemon_PYR = fulldict['spikemon_PYR']
    spikemon_PV = fulldict['spikemon_PV']
    spikemon_SOM = fulldict['spikemon_SOM']

    som_opto = fulldict['som_opto']
    pv_opto = fulldict['pv_opto']

    stim_arr = fulldict['stim_arr']
    stim_dt = fulldict['stim_dt']

    T = fulldict['T']

    if choice == 'traces':
        fig = plt.figure()
        ax11 = fig.add_subplot(411)
        ax21 = fig.add_subplot(412)
        ax31 = fig.add_subplot(413)
        ax41 = fig.add_subplot(414)

        ax11.plot(M_PYR.t / ms, M_PYR.v[0] / mV, label='PYR')
        ax11.plot(M_PYR.t / ms, M_PYR.vD[0] / mV, label='Dend')
        ax11.plot(M_PV.t / ms, M_PV.v[0] / mV, label='PV')

        ax21.plot(M_PYR.t / ms, M_PYR.w[0] / nA, label='PYR Adap.')
        ax21.plot(M_PYR.t / ms, M_PYR.ges[0] / nS, label='inc SOM conduc')
        ax21.plot(M_PYR.t / ms, M_PYR.DSlow[0], label='PYR Dep.')
        ax21.plot(M_PYR.t / ms, M_PYR.DFast[0], label='PYR Dep. fast')
        ax21.plot(M_PYR.t / ms, M_PYR.IFast[0], label='PYR I fast')

        ax21.plot(M_PV.t / ms, M_PV.DSlow[0], label='PV DSlow')
        ax21.plot(M_PV.t / ms, M_PV.DFast[0], label='PV DFast')
        ax21.legend()

        ax31.plot(M_SOM.t / ms, M_SOM.v[0] / mV, label='SOM')

        # get spike times of neuron 0
        idx = np.where(spikemon_PYR.i == 0)[0]
        ax11.scatter(spikemon_PYR.t[idx] / ms, spikemon_PYR.i[idx] + 1, color='k')

        ax11.legend()

    elif choice == 'rasters':

        # plot rasters
        fig = plt.figure()
        gs = gridspec.GridSpec(6, 1)
        ax1 = plt.subplot(gs[:4, :])
        ax2 = plt.subplot(gs[4, :])
        ax3 = plt.subplot(gs[5, :])

        ax1.scatter(spikemon_PYR.t / ms, spikemon_PYR.i, color='k', s=.1)
        ax2.scatter(spikemon_PV.t / ms, spikemon_PV.i, color='k', s=.1)
        ax3.scatter(spikemon_SOM.t / ms, spikemon_SOM.i, color='k', s=.1)

        ax1.set_xlim(0, T / ms)
        ax2.set_xlim(0, T / ms)
        ax3.set_xlim(0, T / ms)

    elif choice == 'psth':

        # plot PSTH
        fig = plt.figure()
        gs = gridspec.GridSpec(3, 1)
        ax31 = plt.subplot(gs[0, :])
        ax32 = plt.subplot(gs[1, :])
        ax33 = plt.subplot(gs[2, :])

        bin_factor = 2. / 2

        ax31.hist(spikemon_PYR.t / ms, color='k', bins=int((T / ms) * bin_factor))
        ax32.hist(spikemon_PV.t / ms, color='k', bins=int((T / ms) * bin_factor))
        ax33.hist(spikemon_SOM.t / ms, color='k', bins=int((T / ms) * bin_factor))

        ax31.set_xlim(0, T / ms)
        ax32.set_xlim(0, T / ms)
        ax33.set_xlim(0, T / ms)

    return fig


def main():
    fulldict1 = setup_and_run()

    if True:
        # plot(fulldict1,choice='rasters')
        plot(fulldict1, choice='traces')
        # plot(fulldict1,choice='psth')
        # plt.show()

    fulldict2 = setup_and_run(pv_opto1=.2)
    fulldict3 = setup_and_run(som_opto1=1)

    fig = plt.figure()
    gs = gridspec.GridSpec(3, 1)

    ax11 = plt.subplot(gs[0, 0])  # psth control
    ax21 = plt.subplot(gs[1, 0])  # psth pvoff
    ax31 = plt.subplot(gs[2, 0])  # psth somoff

    bin_factor = 1. / 2

    ax11.hist(fulldict1['spikemon_PYR'].t / ms, color='k', bins=int((fulldict1['T'] / ms) * bin_factor))
    ax21.hist(fulldict2['spikemon_PYR'].t / ms, color='k', bins=int((fulldict2['T'] / ms) * bin_factor))
    ax31.hist(fulldict3['spikemon_PYR'].t / ms, color='k', bins=int((fulldict3['T'] / ms) * bin_factor))

    ax11.set_title('Pyr Activity (Control)')
    ax21.set_title('PV Activity (PV Off)')
    ax31.set_title('SOM Activity (SOM Off)')

    start_time1 = 160
    start_time2 = 230
    interval_time = 50

    # 50-60...90-100,  start_time2-130...160-170    
    control_pa = get_FR(start_time1, start_time1 + interval_time, fulldict1['defaultclock'],
                        fulldict1['spikemon_PYR'].t)
    control_fs = get_FR(start_time2, start_time2 + interval_time, fulldict1['defaultclock'],
                        fulldict1['spikemon_PYR'].t)

    pv_pa = get_FR(start_time1, start_time1 + interval_time, fulldict2['defaultclock'], fulldict2['spikemon_PYR'].t)
    pv_fs = get_FR(start_time2, start_time2 + interval_time, fulldict2['defaultclock'], fulldict2['spikemon_PYR'].t)

    som_pa = get_FR(start_time1, start_time1 + interval_time, fulldict3['defaultclock'], fulldict3['spikemon_PYR'].t)
    som_fs = get_FR(start_time2, start_time2 + interval_time, fulldict3['defaultclock'], fulldict3['spikemon_PYR'].t)

    print('control:', control_fs / control_pa)
    print('pv off:', pv_fs / pv_pa)
    print('som off:', som_fs / som_pa)

    """
    FR_control_pyr = collect_FR(fulldict1['stim_arr'],
                                fulldict1['stim_dt'],
                                fulldict1['defaultclock'],
                                fulldict1['spikemon_PYR'].t)

    FR_control_pv = collect_FR(fulldict1['stim_arr'],
                               fulldict1['stim_dt'],
                               fulldict1['defaultclock'],
                               fulldict1['spikemon_PV'].t)

    FR_control_som = collect_FR(fulldict1['stim_arr'],
                                fulldict1['stim_dt'],
                                fulldict1['defaultclock'],
                                fulldict1['spikemon_SOM'].t)

    FR_pvoff_pyr = collect_FR(fulldict2['stim_arr'],
                                fulldict2['stim_dt'],
                                fulldict2['defaultclock'],
                                fulldict2['spikemon_PYR'].t)

    FR_pvoff_pv = collect_FR(fulldict2['stim_arr'],
                               fulldict2['stim_dt'],
                               fulldict2['defaultclock'],
                               fulldict2['spikemon_PV'].t)

    FR_pvoff_som = collect_FR(fulldict2['stim_arr'],
                               fulldict2['stim_dt'],
                               fulldict2['defaultclock'],
                               fulldict2['spikemon_SOM'].t)

    FR_somoff_pyr = collect_FR(fulldict3['stim_arr'],
                                fulldict3['stim_dt'],
                                fulldict3['defaultclock'],
                                fulldict3['spikemon_PYR'].t)

    FR_somoff_pv = collect_FR(fulldict3['stim_arr'],
                               fulldict3['stim_dt'],
                               fulldict3['defaultclock'],
                               fulldict3['spikemon_PV'].t)

    FR_somoff_som = collect_FR(fulldict3['stim_arr'],
                               fulldict3['stim_dt'],
                               fulldict3['defaultclock'],
                               fulldict3['spikemon_SOM'].t)

    adapted_fr = FR_control_pyr[-1]


    fig = plt.figure()
    gs = gridspec.GridSpec(3,3)

    ax11 = plt.subplot(gs[0,0]) # psth control
    ax21 = plt.subplot(gs[1,0]) # psth pvoff
    ax31 = plt.subplot(gs[2,0]) # psth somoff

    ax12 = plt.subplot(gs[:,1]) # plot FR
    ax13 = plt.subplot(gs[:,2]) # plot FR diff.


    bin_factor = 1./2

    ax11.hist(fulldict1['spikemon_PYR'].t/ms, color='k',bins=int((fulldict1['T']/ms)*bin_factor))
    #ax11.hist(fulldict1['spikemon_PYR'].t/ms, color='k',bins=int((fulldict1['T']/ms)*bin_factor))

    ax21.hist(fulldict2['spikemon_PYR'].t/ms, color='k',bins=int((fulldict2['T']/ms)*bin_factor))
    ax31.hist(fulldict3['spikemon_PYR'].t/ms, color='k',bins=int((fulldict3['T']/ms)*bin_factor))

    ax11.set_xlim(500,fulldict1['T']/ms)
    ax21.set_xlim(500,fulldict2['T']/ms)
    ax31.set_xlim(500,fulldict3['T']/ms)


    tone_number = np.arange(len(np.where(stim_arr!=0)[0]))
    bar_width = 0.2
    ax12.set_title('Mean FR')
    ax12.bar(tone_number,FR_control_pyr/adapted_fr,width=bar_width,label='control',color='blue')
    ax12.bar(tone_number+bar_width,FR_pvoff_pyr/adapted_fr,width=bar_width,label='pv_off',color='green')
    ax12.bar(tone_number+2*bar_width,FR_somoff_pyr/adapted_fr,width=bar_width,label='som_off',color='red')
    ax12.plot([0,4],[1,1],ls='--',color='gray')


    #ax13 = plt.subplot(gs[:,2])

    ax13.set_title('Diff from Control')
    ax13.bar(tone_number,np.abs(FR_control_pyr-FR_pvoff_pyr)/adapted_fr,
             width=bar_width,label='control-pv_off',color='green')
    ax13.bar(tone_number+bar_width,np.abs(FR_control_pyr-FR_somoff_pyr)/adapted_fr,
             width=bar_width,label='control-som_off',color='red')

    #print(collect_FR(stim_arr,stim_dt,defaultclock.dt,spikemon_PV.t),'PV')
    #print(collect_FR(stim_arr,stim_dt,defaultclock.dt,spikemon_SOM.t),'SOM')



    # plot mean FR
    #fig3 = plt.figure()
    """

    plt.show()


if __name__ == "__main__":
    main()