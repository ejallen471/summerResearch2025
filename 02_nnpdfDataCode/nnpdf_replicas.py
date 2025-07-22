"""
 LHAPDF uses the Particle Data Group (PDG) numbering scheme
 to enumerate particle species. The full list is available here:
   https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf
 For our purpose, we only need parton flavours, which are:
 21:   gluon
 1:    down
 2:    up
 3:    strange
 4:    charm
 5:    bottom
 6:    top
 Anti-particle (with exception of the gluon of course) are numbered using
 negative numbers (-1 for anti-down, -2 for anti-up, etc...). This ordering
 is known as flavour basis.

 In fitting PDF, one usally rotates the flavour basis into the so-called
 evolution basis. The reeason is that this basis diagonalises the evolution
 equations (knwon as DGLAP equations) needed to determine PDFs from data.
 Each element of the evolution basis is a linear combination of the flavours
 above. Note that this map is not unique, and there might be different definitions
 of evolution basis. In NNPDF, we use the definition shown at this link:
    https://eko.readthedocs.io/en/latest/theory/FlavorSpace.html#qcd-evolution

Since LHAPDF delivers PDF sets in flavour basis, we will use this map
to rotate flavours into the evolution basis.
"""

import numpy as np
import lhapdf as lh
import matplotlib.pyplot as plt
from  matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text',usetex=True)

import pickle
from pathlib import Path

SERIALIZATION_DIR = Path("./")
FLAV_PATH = SERIALIZATION_DIR / "flavour_basis.pkl"
EV_PATH = SERIALIZATION_DIR / "evolution_basis.pkl"

# where is XGRID from?
XGRID = np.array(
      [2.00000000e-07, 3.03430477e-07, 4.60350147e-07, 6.98420853e-07,
       1.05960950e-06, 1.60758550e-06, 2.43894329e-06, 3.70022721e-06,
       5.61375772e-06, 8.51680668e-06, 1.29210157e-05, 1.96025050e-05,
       2.97384954e-05, 4.51143839e-05, 6.84374492e-05, 1.03811730e-04,
       1.57456056e-04, 2.38787829e-04, 3.62054496e-04, 5.48779532e-04,
       8.31406884e-04, 1.25867971e-03, 1.90346340e-03, 2.87386758e-03,
       4.32850064e-03, 6.49620619e-03, 9.69915957e-03, 1.43750686e-02,
       2.10891867e-02, 3.05215840e-02, 4.34149174e-02, 6.04800288e-02,
       8.22812213e-02, 1.09143757e-01, 1.41120806e-01, 1.78025660e-01,
       2.19504127e-01, 2.65113704e-01, 3.14387401e-01, 3.66875319e-01,
       4.22166775e-01, 4.79898903e-01, 5.39757234e-01, 6.01472198e-01,
       6.64813948e-01, 7.29586844e-01, 7.95624252e-01, 8.62783932e-01,
       9.30944081e-01, 1.00000000e+00])

FLAV_TO_EV_MAP = [
        {'u': 1, 'ubar':  1, 'd':  1, 'dbar':  1, 's':  1, 'sbar':  1, 'c': 2, 'g': 0 }, # Sigma
        {'u': 1, 'ubar': -1, 'd':  1, 'dbar': -1, 's':  1, 'sbar': -1, 'c': 0, 'g': 0 }, # V
        {'u': 1, 'ubar': -1, 'd': -1, 'dbar':  1, 's':  0, 'sbar':  0, 'c': 0, 'g': 0 }, # V3
        {'u': 1, 'ubar': -1, 'd':  1, 'dbar': -1, 's': -2, 'sbar':  2, 'c': 0, 'g': 0 }, # V8
        {'u': 1, 'ubar':  1, 'd': -1, 'dbar': -1, 's':  0, 'sbar':  0, 'c': 0, 'g': 0 }, # T3 
        {'u': 1, 'ubar':  1, 'd':  1, 'dbar':  1, 's': -2, 'sbar': -2, 'c': 0, 'g': 0 }, # T8
        {'u': 0, 'ubar':  0, 'd':  0, 'dbar':  0, 's':  0, 'sbar':  0, 'c': 2, 'g': 0 }, # c+
        {'u': 0, 'ubar':  0, 'd':  0, 'dbar':  0, 's':  0, 'sbar':  0, 'c': 0, 'g': 1 }, # g
        {'u': 1, 'ubar': -1, 'd':  1, 'dbar': -1, 's':  1, 'sbar': -1, 'c': 0, 'g': 0 }, # V15      
        ]

LABELS_EV = [
    "Sigma",
    "V",
    "V3",
    "V8",
    "T3",
    "T8",
    "c+",
    "g",
    "V15",]

LABELS_LATEX = [
        r"$\Sigma$",
        r"$V$",
        r"$V_3$",
        r"$V_8$",
        r"$T_3$",
        r"$T_8$",
        r"$c^+$",
        r"$g$",
        r"$V_{15}$",
        ]

PID_FLAVS = {
        1: "d",
        2: "u",
        3: "s",
        4: "c",
        -1: "dbar",
        -2: "ubar",
        -3: "sbar",
        -4: "cbar",
        21: "g",
        }

# PDFs depend on a scale
Q = 1.65 # GeV

def evaluate_replicas(pdf_set):
    
    res = []
    for rep in pdf_set:
        flav_dict = {}
        for pid, key in PID_FLAVS.items():
            tmp = np.zeros(shape=(XGRID.size,))
            for idx, x in enumerate(XGRID):
                tmp[idx] = rep.xfxQ(pid, x, Q)
            
            flav_dict[key] = tmp
        
        res.append(flav_dict)

    return res

def rotate_to_ev(pdf_flav):
    res = []
    for rep in pdf_flav:
        ev_dict = {}
        for map, label in zip(FLAV_TO_EV_MAP, LABELS_EV):
            tmp = np.zeros(shape=(XGRID.size,))
            for idx, x in enumerate(XGRID):
                for k,v in map.items():
                    tmp[idx] += v * rep[k][idx]

            ev_dict[label] = tmp

        res.append(ev_dict)

    return res

def compute_mean_and_std(pdf_set, flav):
    tmp = np.empty((len(pdf_set), len(pdf_set[0][flav])))
    for idx, rep in enumerate(pdf_set):
        tmp[idx] = rep[flav]
    
    mean = tmp.mean(axis=0)
    std = tmp.std(axis=0)

    return mean, std

def plot_flavs(pdf_flav):
    
    return fig, axs


def main():
    if FLAV_PATH.exists() and EV_PATH.exists():
        print("Loading existing data...")

        with open(FLAV_PATH, 'rb') as f:
            res_flav = pickle.load(f)
        with open(EV_PATH, 'rb') as f:
            res_ev = pickle.load(f)
    
    else:
        print("Computing data...")
        # Load pdf set
        pdf_set = lh.mkPDFs("NNPDF40_nnlo_as_01180_1000")

        # Remove central replica
        _ = pdf_set.pop(0)

        res_flav = evaluate_replicas(pdf_set)
        res_ev = rotate_to_ev(res_flav)

        # Serialize data
        with open('flavour_basis.pkl', 'wb') as f:
            pickle.dump(res_flav, f)

        with open('evolution_basis.pkl', 'wb') as f:
            pickle.dump(res_ev, f)
        
    
    # =========== PLOTS IN EVOLUTION BASIS =================
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('PDFs in evolution basis', fontsize=16, fontweight='bold')
    flav_order = ['u', 'd', 's', 'ubar', 'dbar', 'sbar', 'c', 'cbar', 'g']
    y_labels = [r'$xu(x)$', r'$xd(x)$', r'$xs(x)$', r'$x \bar{u}(x)$', r'$x \bar{d}(x)$', r'$x \bar{s}(x)$', r'$xc$', r'$x \bar{c}(x)$', r'$xg$']
    y_lims = [(0.35, 0.80), (0.30, 0.6), (0.0, 0.55), (0.0, 0.55), (0.0, 0.55), (0.0, 0.55), (-0.06, 0.03), (-0.06, 0.03), (0.5, 3.5)]
    axes_flat = axes.flatten()

    for ax, flav, y_label, y_lim in zip(axes_flat, flav_order, y_labels, y_lims):
        # Sample plot data
        mean, std = compute_mean_and_std(res_flav, flav)
        ax.plot(XGRID, mean, linewidth=2, label=r'$\textrm{mean} \pm \sigma$')
        ax.fill_between(XGRID, mean - std, mean + std, alpha=0.4)
        
        # Customize subplot
        ax.set_ylabel(y_label, fontsize=20)
        ax.set_xlabel(r'$x$')
        ax.set_xscale('log')
        ax.set_xlim([1e-5,1])
        ax.set_ylim(y_lim)

    plt.show()

    # =========== PLOTS IN FLAVOUR BASIS =================
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('PDFs in flavour basis', fontsize=16, fontweight='bold')
    flav_order = LABELS_EV
    y_labels = LABELS_LATEX
    axes_flat = axes.flatten()

    for ax, flav, y_label in zip(axes_flat, flav_order, y_labels):
        # Sample plot data
        mean, std = compute_mean_and_std(res_ev, flav)
        ax.plot(XGRID, mean, linewidth=2, label=r'$\textrm{mean} \pm \sigma$')
        ax.fill_between(XGRID, mean - std, mean + std, alpha=0.4)
        
        # Customize subplot
        ax.set_ylabel(y_label, fontsize=20)
        ax.set_xlabel(r'$x$')
        ax.set_xscale('log')
        ax.set_xlim([1e-5,1])
        ax.legend()

    plt.show()

if __name__ == "__main__":
    main()
