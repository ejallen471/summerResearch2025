"""
We have the original NNPDF4.0 LHAPDF set and want reference mean and standard
deviation values for comparison with the KDE reconstruction.

Run with the following command:

python generate_reference_statistics.py

This file does the following:

1. Load all original NNPDF4.0 uncertainty members through LHAPDF.
2. Evaluate the required flavours on the 45-point x-grid at every Q and rotate
   values into the evolution basis when required.
3. Save per-flavour reference mean and standard-deviation CSV files and plots.
"""

import numpy as np
import lhapdf as lh
import matplotlib.pyplot as plt
from  matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text',usetex=True)
import os

import pickle
from pathlib import Path

plt.style.use('../pythonStyle.mplstyle')

SERIALIZATION_DIR = Path("./")
FLAV_PATH = SERIALIZATION_DIR / "flavour_basis.pkl"
EV_PATH = SERIALIZATION_DIR / "evolution_basis.pkl"

# original
# XGRID = np.array(
#       [2.00000000e-07, 3.03430477e-07, 4.60350147e-07, 6.98420853e-07,
#        1.05960950e-06, 1.60758550e-06, 2.43894329e-06, 3.70022721e-06,
#        5.61375772e-06, 8.51680668e-06, 1.29210157e-05, 1.96025050e-05,
#        2.97384954e-05, 4.51143839e-05, 6.84374492e-05, 1.03811730e-04,
#        1.57456056e-04, 2.38787829e-04, 3.62054496e-04, 5.48779532e-04,
#        8.31406884e-04, 1.25867971e-03, 1.90346340e-03, 2.87386758e-03,
#        4.32850064e-03, 6.49620619e-03, 9.69915957e-03, 1.43750686e-02,
#        2.10891867e-02, 3.05215840e-02, 4.34149174e-02, 6.04800288e-02,
#        8.22812213e-02, 1.09143757e-01, 1.41120806e-01, 1.78025660e-01,
#        2.19504127e-01, 2.65113704e-01, 3.14387401e-01, 3.66875319e-01,
#        4.22166775e-01, 4.79898903e-01, 5.39757234e-01, 6.01472198e-01,
#        6.64813948e-01, 7.29586844e-01, 7.95624252e-01, 8.62783932e-01,
#        9.30944081e-01, 1.00000000e+00])

# 45 Grid (as run with the KDE reconstruction)
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
       6.64813948e-01])

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


def evaluate_replicas(pdf_set, Q):
    
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


FLAV_ORDER = ['d', 'u', 's', 'c', 'dbar', 'ubar', 'sbar', 'cbar', 'g']

def save_statistics(Q, res_flav):

    os.makedirs("mean", exist_ok=True)
    os.makedirs("std", exist_ok=True)

    for flav in FLAV_ORDER:
        mean, std = compute_mean_and_std(res_flav, flav)

        mean_filename = f"mean/mean_{flav}_Q={Q:.6e}.csv"
        std_filename = f"std/std_{flav}_Q={Q:.6e}.csv"

        np.savetxt(mean_filename, mean, delimiter=",", fmt="%.6e")
        np.savetxt(std_filename, std, delimiter=",", fmt="%.6e")

    print(f"Saved mean & std CSV files for Q={Q:.6e}")

def main():

    Q_values = [1.6500000e+00, 1.7874388e+00, 1.9429053e+00, 2.1193749e+00, 2.3204100e+00,
            2.5502944e+00, 2.8142025e+00, 3.1184122e+00, 3.4705775e+00, 3.8800751e+00, 4.3584516e+00,
            4.9200000e+00, 4.9200000e+00, 5.5493622e+00, 6.2897452e+00, 7.1650687e+00, 8.2052867e+00,
            9.4481248e+00, 1.0941378e+01, 1.2745972e+01, 1.4940062e+01, 1.7624572e+01, 2.0930715e+01,
            2.5030298e+01, 3.0149928e+01, 3.6590777e+01, 4.4756282e+01, 5.5191298e+01, 6.8637940e+01,
            8.6115921e+01, 1.0903923e+02, 1.3938725e+02, 1.7995815e+02, 2.3474820e+02, 3.0952544e+02,
            4.1270732e+02, 5.5671861e+02, 7.6011795e+02, 1.0509694e+03, 1.4722574e+03, 2.0906996e+03,
            3.0112909e+03, 4.4016501e+03, 6.5333918e+03, 9.8535186e+03, 1.5109614e+04, 2.3573066e+04,
            3.7444017e+04, 6.0599320e+04, 1.0000000e+05]

    for Q in Q_values:
    
        print("Computing data...")
        # Load pdf set
        # pdf_set = lh.mkPDFs("NNPDF31_nnlo_as_0118") # 3.1 version nnpdf
        pdf_set = lh.mkPDFs("NNPDF_original")

        # Remove central replica
        _ = pdf_set.pop(0)

        res_flav = evaluate_replicas(pdf_set, Q)
        res_ev = rotate_to_ev(res_flav)

        # # Serialize data
        # with open('flavour_basis.pkl', 'wb') as f:
        #     pickle.dump(res_flav, f)

        # with open('evolution_basis.pkl', 'wb') as f:
        #     pickle.dump(res_ev, f)

        os.makedirs("mean", exist_ok=True)
        os.makedirs("std", exist_ok=True)

        save_statistics(Q, res_flav)

        print("All mean/std files saved successfully!")
        
    
    # =========== PLOTS IN FLAVOUR BASIS =================

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('PDFs in Flavour Basis', fontsize=16, fontweight='bold')
        flav_order = FLAV_ORDER
        y_labels = [r'$xd(x)$', r'$xu(x)$', r'$xs(x)$', r'$xc$', r'$x \bar{d}(x)$', r'$x \bar{u}(x)$', r'$x \bar{s}(x)$', r'$x \bar{c}(x)$', r'$xg$']
        # y_labels = [r'$d$', r'$u$', r'$s$', r'$c$', r'$\bar{d}$', r'$\bar{u}$', r'$\bar{s}$', r'$\bar{c}$', r'$g$']
        y_lims = [(0.30, 0.6), (0.35, 0.80), (0.0, 0.55), (-0.06, 0.15), (0.0, 0.55), (0.0, 0.55), (0.0, 0.55), (-0.06, 0.15), (0.5, 3.5)]
        
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
        
        plt.savefig(f"PDF_lhapdf_{Q}.png")
        # plt.show()

if __name__ == "__main__":
    main()
