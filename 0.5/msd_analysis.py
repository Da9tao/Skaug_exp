import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.optimize import curve_fit
import os

# 繪圖設定
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 10,
    'figure.figsize': (7, 5),
})

def fit_func_FFPE(t, K, a):
    return 4 * K * t**a / gamma(1 + a)

def calculate_r_squared(y_true, y_pred):
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    total_sum_of_squares = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r_squared

def ref_msd(Dp0, beta, dt):
    dt *= 0.1
    t = np.arange(1, 5, dt)
    msd = 4 * Dp0 * t ** beta
    return t, msd

def k_Gamma_function(l, t, bins=50, show_plot=False):
    l = np.asarray(l)
    mean_l = np.mean(l)
    std_l = np.std(l)
    k = (mean_l / std_l) ** 2

    # 計算 Gamma distribution fit (PDF)
    coef = (k ** k) / gamma(k)
    l_term = l ** (k - 1)
    mean_term = mean_l ** k
    exp_term = np.exp(-k * l / mean_l)
    fun = coef * l_term / mean_term * exp_term
    return mean_l, k, fun

def msd_mean(t, time, msd_total, valid_batch):
    msd_total /= valid_batch
    
    ax1.scatter(time[::100], msd_total[::100], s=3, label=f'$t_{{form}}={t}$')
    ax1.legend()
    
    popt, _  = curve_fit(fit_func_FFPE, time, msd_total)
    fit_K, fit_a = popt
    fit_msd = fit_func_FFPE(time, fit_K, fit_a)
    r2 = calculate_r_squared(msd_total, fit_msd)
    print(f't={t}, alpha={fit_a:.4f}, Ka={fit_K:.4e}, R2={r2:.4f}')

t_to_size = {
    1000:   247.1080942,
    2000:   216.5761537,
    3000:   204.049624,
    4000:   191.427262,
    5000:   184.5860421,
    6000:   174.8961444,
    7000:   169.1660005,
    8000:   166.2407307,
    9000:   161.038961,
    10000:  156.051519,
    30000:  120.5969061,
    50000:  105.975122,
    70000:  98.08307894,
    100000: 87.42639647,
    200000: 72.80066049,
    400000: 59.41720823,
    700000: 50.42732327
}
L = 256
t_values = [1000]

particle_size = 100              
L_sim = 256
D0_sim = 1.0
D0_ref = {40:0.088, 100:0.037, 200:0.02}   #um^2/s
Dp0_ref = {40:0.044, 100:0.019, 200:0.008}
beta_ref = {40:0.83, 100:0.77,  200:0.85}
dt = 0


fig1, ax1 = plt.subplots()
current_dir = os.getcwd()
os.chdir(f'{particle_size}nm_output')
for tf in t_values:
    l = t_to_size[tf] / L_sim    #um
    
    D_scale = D0_ref[particle_size]
    t_scale = l**2/D0_ref[particle_size]
    print(f'tf={tf} D_scale={D_scale}  t_unit={t_scale}')

    alpha_batch = []
    Ka_batch = []
    r2_batch = []
    msd_total = None
    valid_batch = 0
    for b in range(10):
        filename = f'dist_chord_msd_dist_map_phi=0.5 t={tf}_batch_{b}.txt'  
        t, msd = np.loadtxt(filename).T
        
        t *= t_scale
        msd *= l**2
        #D = msd  / (4 * t )
        #plt.plot(t, D, label=fr'$t_{{\mathrm{{form}}}}={tf}$', marker="o",markersize=0.1)
        
        #e = int(6/t[1])
        #t = t[:e]
        #msd = msd[:e]
        if msd_total is None:
            msd_total = np.zeros_like(msd)
        popt, _ = curve_fit(fit_func_FFPE, t, msd)
        Ka, alpha = popt
        msd_fit = fit_func_FFPE(t, Ka, alpha)
        r2 = calculate_r_squared(msd, msd_fit)

        alpha_batch.append(alpha)
        Ka_batch.append(Ka)
        r2_batch.append(r2)
        msd_total += msd
        valid_batch += 1
    D = msd_total  / (4 * t )
    #D *= D_scale

    #plt.plot(t[s:e], D[s:e], label=fr'$t_{{\mathrm{{form}}}}={t}\ D_0^{{sim}}={D0_sim}$', marker="o",markersize=0.1)
    plt.plot(t, D, label=fr'$t_{{\mathrm{{form}}}}={tf}$', marker="o",markersize=0.1)
    popt, pcov = curve_fit(fit_func_FFPE, t, msd_total)
    K ,a = popt
    msd_fit = fit_func_FFPE(t, K, a)
    r2 = calculate_r_squared(msd_total, msd_fit)
    print(f'K={K} and a={a} and r2={r2}')

'''
# 加入實驗資料
try:
    exp_data = np.loadtxt(f'D_digitizer_{particle_size}nm_raw.txt').T
    plt.scatter(exp_data[0], exp_data[1], color='r', s=10, label=f'Digitized Exp ({particle_size} nm).')
except Exception as e:
    print("Experimental data file not found or unreadable:", e)
'''
exp_t, exp_msd = ref_msd(Dp0_ref[particle_size], beta_ref[particle_size], 0.001)
exp_data = exp_msd / (4*exp_t)
plt.scatter(exp_t, exp_data, color='r', s=10, label=f'Exp ({particle_size} nm).')
    


plt.xlabel(r'$t\ (s)$')
plt.ylabel(r'$D(t)\ (\mu\mathrm{m}^2/s)$')
#plt.xscale('log')
#plt.yscale('log')
plt.xlim(1,5)
plt.legend(loc="upper right", ncol=1)
plt.tight_layout()
plt.savefig(f'msd_dist_map_phi=0.5_{particle_size}nm_D_t.png', dpi=300)
plt.close()
