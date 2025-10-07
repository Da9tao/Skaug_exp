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
    coef = (k ** k) / gamma(k)
    l_term = l ** (k - 1)
    mean_term = mean_l ** k
    exp_term = np.exp(-k * l / mean_l)
    fun = coef * l_term / mean_term * exp_term
    return mean_l, k, fun

# ===== 參數 =====
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

particle_size = 200
L_sim = 256
D0_sim = 1.0
D0_ref = {40:0.088, 100:0.037, 200:0.02}   # um^2/s
Dp0_ref = {40:0.044, 100:0.019, 200:0.008}
beta_ref = {40:0.83, 100:0.77,  200:0.85}

# ===== 主流程 =====
fig1, ax1 = plt.subplots()
current_dir = os.getcwd()

def process_dir(dir_name, label_suffix=""):
    os.chdir(dir_name)
    for tf in t_values:
        l = t_to_size[tf] / L_sim    # um
        D_scale = D0_ref[particle_size]
        t_scale = l**2 / D0_ref[particle_size]
        print(f'tf={tf} D_scale={D_scale}  t_unit={t_scale}')

        alpha_batch, Ka_batch, r2_batch = [], [], []
        t_ref = None
        msd_sum = None
        valid_batch = 0

        for b in range(10):
            filename = f'dist_chord_msd_dist_map_phi=0.5 t={tf}_batch_{b}.txt'
            t, msd = np.loadtxt(filename).T

            # 單位轉換
            t = t * t_scale
            msd = msd * (l**2)

            # 避免 t=0（導數與 D 比值都會不穩）
            mask = t > 0
            if not np.any(mask):
                continue
            t = t[mask]
            msd = msd[mask]

            # 建立/對齊共同時間軸
            if t_ref is None:
                t_ref = t.copy()
                msd_sum = np.zeros_like(t_ref)
            else:
                # 若這批 t 跟 t_ref 不同，插值到 t_ref
                if (t.shape != t_ref.shape) or np.max(np.abs(t - t_ref)) > 1e-12:
                    msd = np.interp(t_ref, t, msd)
                t = t_ref

            # 擬合（可選，保留你原本的輸出）
            try:
                popt, _ = curve_fit(fit_func_FFPE, t, msd,
                                    p0=(msd[1]/(4*t[1]), 1.0),
                                    bounds=([0.0, 0.0], [np.inf, 2.0]))
                Ka, alpha = popt
                msd_fit = fit_func_FFPE(t, Ka, alpha)
                r2 = calculate_r_squared(msd, msd_fit)
                alpha_batch.append(alpha); Ka_batch.append(Ka); r2_batch.append(r2)
            except Exception as e:
                print(f"Batch {b} fit failed: {e}")

            msd_sum += msd
            valid_batch += 1

        if valid_batch == 0:
            print(f"No valid batches for tf={tf} in {dir_name}")
            continue

        # 平均 MSD
        msd_mean = msd_sum / valid_batch

        # ===== 以導數法計算 D(t) =====
        # 中心差分（np.gradient 對內部點為中心差分，邊界為單邊差分）
        dmsd_dt = np.gradient(msd_mean, t_ref)
        D_t = 0.25 * dmsd_dt  # 2D: 1/(2d) = 1/4

        # 繪圖
        plt.plot(t_ref, D_t, label=fr'$t_{{\mathrm{{form}}}}={tf}$' + label_suffix, marker="o", markersize=0.1)

        # 若你仍想要在 MSD 上做 FFPE 擬合與評估 R^2，可保留
        try:
            popt, pcov = curve_fit(fit_func_FFPE, t_ref, msd_mean)
            K, a = popt
            msd_fit = fit_func_FFPE(t_ref, K, a)
            r2 = calculate_r_squared(msd_mean, msd_fit)
            print(f'tf={tf} K={K:.4e} a={a:.4f} R2={r2:.4f} (dir={dir_name})')
        except Exception as e:
            print(f"Final fit failed for tf={tf} in {dir_name}: {e}")

    os.chdir(current_dir)

# 跑兩個資料夾
process_dir(f'{particle_size}nm_output_ori', label_suffix="")
process_dir(f'{particle_size}nm_output', label_suffix="$_De")

# 加入實驗參考（保持你原本邏輯）
exp_t, exp_msd = ref_msd(Dp0_ref[particle_size], beta_ref[particle_size], 0.001)
exp_data = exp_msd / (4*exp_t)  # 這是用比值法生成的參考 D(t)，可視為視覺參考
plt.scatter(exp_t, exp_data, s=10, label=f'Exp ({particle_size} nm).')

plt.xlabel(r'$t\ (s)$')
plt.ylabel(r'$D(t)\ (\mu\mathrm{m}^2/s)$')
#plt.xscale('log')
#plt.yscale('log')
plt.xlim(1, 5)
plt.legend(loc="upper right", ncol=1)
plt.tight_layout()
plt.savefig(f'msd_dist_map_phi=0.5_{particle_size}nm_D_t_derivative.png', dpi=300)
plt.close()
