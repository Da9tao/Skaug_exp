#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <functional>
using namespace std;
namespace fs = std::filesystem;

const string folder = "./0.5";
const int total_simulations = 10000;
const int N = 1000;
const int num_batches = total_simulations / N;
const int tracer = 10;
const double N_inv = 1.0 / N;
const double tracer_inv = 1.0 / tracer;

const double k = 0.0;
const double L = 256.0;
const double L_inv = 1.0 / L;
const double mu = 0;
const double sigma = 1;
const int rows = L, cols = L;
const int particle_size = 200; // nm
const double R_ref = particle_size * 0.5; // nm

inline int wrapi(int a, int n) {
    a %= n;
    return (a < 0) ? (a + n) : a;
}

// Compute a morphology-specific time step that respects the relaxation
// time-scale constraint dt < (R_ref^2 / D0_ref / 10) * t_scale.
double compute_dt(double t_scale, double D0_ref) {
    const double R_ref_um = 0.001 * R_ref;
    const double relaxation_time = 0.01 * (R_ref_um * R_ref_um) / D0_ref; // s
    const double dt_limit = relaxation_time * t_scale;
    const double dt_candidate = min(0.001, dt_limit);

    if (dt_limit > 0.0 && dt_candidate >= dt_limit) {
        return max(dt_limit, 1e-8);
    }
    return max(dt_candidate, 1e-8);
}

map<int, double> t_to_size = {
    {1000,   247.1080942},
    {2000,   216.5761537},
    {3000,   204.049624},
    {4000,   191.427262},
    {5000,   184.5860421},
    {6000,   174.8961444},
    {7000,   169.1660005},
    {8000,   166.2407307},
    {9000,   161.038961},
    {10000,  156.051519},
    {30000,  120.5969061},
    {50000,  105.975122},
    {70000,  98.08307894},
    {100000, 87.42639647},
    {200000, 72.80066049},
    {400000, 59.41720823},
    {700000, 50.42732327}
};

mutex msd_mutex;
mutex mtx;
condition_variable cv;
queue<int> tasks;
bool done = false;

int get_time(const string& filename) {
    size_t pos1 = filename.find("t=");
    size_t pos2 = filename.find(".txt");
    return stoi(filename.substr(pos1 + 2, pos2 - pos1 - 2));
}

// 讀 dist 並計算 Dx/Dy map（含週期 Sobel 與 De 投影修正）
void morphology_read(const string& filepath,
                     array<array<double, cols>, rows>& dist_map,
                     array<array<double, cols>, rows>& de_x_map,
                     array<array<double, cols>, rows>& de_y_map,
                     double r_ref) {
    vector<double> distances(rows * cols, 0.0);
    ifstream ifs(filepath);
    if (!ifs.is_open()) {
        throw runtime_error("Failed to open morphology file: " + filepath);
    }

    size_t index = 0;
    double value;
    while (ifs >> value && index < distances.size()) {
        distances[index++] = value;
    }
    ifs.close();

    if (index != distances.size()) {
        throw runtime_error("Unexpected morphology size in " + filepath);
    }

    auto idx = [=](size_t i, size_t j) { return i * cols + j; };

    // 寫入距離場到 dist_map
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const double dist = distances[idx(i, j)];
            dist_map[i][j] = dist;
        }
    }

    // ---- Sobel kernels（週期）----
    static constexpr int kx[3][3] = {
        { 1,  0, -1},
        { 2,  0, -2},
        { 1,  0, -1},
    };
    static constexpr int ky[3][3] = {
        { 1,  2,  1},
        { 0,  0,  0},
        {-1, -2, -1},
    };
    const double sobel_scale = 1.0 / 8.0;

    vector<double> grad_x(distances.size(), 0.0);
    vector<double> grad_y(distances.size(), 0.0);

    // ★ 週期邊界：用 wrapi() 索引
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            double gx = 0.0;
            double gy = 0.0;

            for (int dr = -1; dr <= 1; ++dr) {
                int rr = wrapi(r + dr, rows);
                for (int dc = -1; dc <= 1; ++dc) {
                    int cc = wrapi(c + dc, cols);
                    double sample = distances[idx(rr, cc)];
                    gx += static_cast<double>(kx[dr + 1][dc + 1]) * sample;
                    gy += static_cast<double>(ky[dr + 1][dc + 1]) * sample;
                }
            }
            grad_x[idx(r, c)] = gx * sobel_scale;
            grad_y[idx(r, c)] = gy * sobel_scale;
        }
    }

    // ---- 建立 Dx, Dy（正確投影；零梯度 fallback）----
    const double D0 = 1.0;   // 若你要帶入真實 D0 也可以外面傳入；這裡維持你的原設定
    const double eps = 1e-12;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const size_t flat = idx(r, c);
            const double dist = distances[flat];

            if (dist <= eps || r_ref <= eps) {
                de_x_map[r][c] = 0.0;
                de_y_map[r][c] = 0.0;
                continue;
            }

            const double R_over_d = r_ref / dist;
            const double d_over_R = dist / r_ref;

            // Dp（法向）兩段式
            double Dp = 0.0;
            if (R_over_d <= 2.0) {
                const double Rz2 = R_over_d * R_over_d;
                const double Rz3 = Rz2 * R_over_d;
                const double Rz4 = Rz3 * R_over_d;
                const double Rz5 = Rz4 * R_over_d;
                Dp = D0 * (1.0 - 0.5625 * R_over_d + 0.125 * Rz3 - 45.0 * Rz4 / 256.0 - Rz5 / 6.0);
            } else {
                const double logz = log(d_over_R);
                const double denom = logz * logz - 4.325 * logz + 1.591;
                if (denom != 0.0) {
                    Dp = -D0 * (2.0 * (logz - 0.9543)) / denom;
                }
            }
            if (!isfinite(Dp) || Dp < 0.0) Dp = 0.0;

            // Dv（切向）
            const double num = 6.0 * d_over_R * d_over_R + 2.0 * d_over_R;
            const double den = 6.0 * d_over_R * d_over_R + 9.0 * d_over_R + 2.0;
            double Dv = (den != 0.0) ? (D0 * (num / den)) : 0.0;
            if (!isfinite(Dv) || Dv < 0.0) Dv = 0.0;

            // 單位法向 (nx, ny) 來自距離場梯度
            const double gx = grad_x[flat];
            const double gy = grad_y[flat];
            const double g2 = gx * gx + gy * gy;

            if (g2 <= eps) {
                // 零梯度：方向不明 → 用 Dv 作為保底（也可用 (Dp+Dv)/2 依偏好）
                de_x_map[r][c] = Dv;
                de_y_map[r][c] = Dv;
                continue;
            }

            const double invg = 1.0 / sqrt(g2);
            const double nx = gx * invg;
            const double ny = gy * invg;

            // ★ 正確的張量投影到座標軸（cos^2 權重）
            const double Dx = Dv + (Dp - Dv) * (nx * nx);
            const double Dy = Dv + (Dp - Dv) * (ny * ny);

            de_x_map[r][c] = std::max(0.0, Dx);
            de_y_map[r][c] = std::max(0.0, Dy);
        }
    }
}

pair<int, int> get_new_coordinates(double x, double y) {
    int xd = static_cast<int>(round(x - L * round((x - L * 0.5) * L_inv))) % rows;
    int yd = static_cast<int>(round(y - L * round((y - L * 0.5) * L_inv))) % rows;
    return {xd, yd};
}

void simulate_trajectory(const array<array<double, cols>, rows>& dist_map,
                         const array<array<double, cols>, rows>& de_x_map,
                         const array<array<double, cols>, rows>& de_y_map,
                         int n0,
                         double dt,
                         vector<double>& msd,
                         double r_ref) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dis(mu, sigma);
    uniform_real_distribution<> dis_uniform(0.0, 1.0);

    vector<double> local_msd(n0, 0.0);
    vector<double> xs0_i(tracer);
    vector<double> ys0_i(tracer);

    // Initialize initial positions
    for (int i = 0; i < tracer; ++i) {
        int xi, yi;
        do {
            double u1 = dis_uniform(gen);
            double u2 = dis_uniform(gen);
            double n1 = L * abs(u1);
            double n2 = L * abs(u2);

            xi = int(n1) % static_cast<int>(L);
            yi = int(n2) % static_cast<int>(L);
        } while (dist_map[xi][yi] == 0.0);

        xs0_i[i] = xi;
        ys0_i[i] = yi;
    }

    // Simulate the trajectory for each tracer
    for (int i = 0; i < tracer; ++i) {
        double xs0 = xs0_i[i], ys0 = ys0_i[i];

        for (int j = 0; j < n0; ++j) {
            double xs1 = xs0, ys1 = ys0;

            if (j > 0) {
                double u3 = dis(gen);
                double u4 = dis(gen);

                auto [xc, yc] = get_new_coordinates(xs0, ys0);
                double local_Dx = de_x_map[xc][yc];
                double local_Dy = de_y_map[xc][yc];

                local_Dx = max(local_Dx, 0.0);
                local_Dy = max(local_Dy, 0.0);

                xs0 += k * dt * (-xs0) + sqrt(2.0 * local_Dx * dt) * u3;
                ys0 += k * dt * (-ys0) + sqrt(2.0 * local_Dy * dt) * u4;

                auto [xd, yd] = get_new_coordinates(xs0, ys0);

                if (dist_map[xd][yd] <= r_ref) {
                    auto [xd1, yd1] = get_new_coordinates(xs1, ys1);
                    if (xd < xd1){
                        xs0 = xs1 + abs(xs1 - xs0);
                    }
                    else if (xd > xd1){
                        xs0 = xs1 - abs(xs1 - xs0);
                    }
                    if (yd < yd1){
                        ys0 = ys1 + abs(ys1 - ys0);
                    }
                    else if (yd > yd1){
                        ys0 = ys1 - abs(ys1 - ys0);
                    }

                    auto [xd_new, yd_new] = get_new_coordinates(xs0, ys0);
                    if (dist_map[xd_new][yd_new] <= r_ref) {
                        xs0 = xs1;
                        ys0 = ys1;
                    }
                }
                double dx = xs0 - xs0_i[i];
                double dy = ys0 - ys0_i[i];
                double dr2 = dx * dx + dy * dy;
                local_msd[j] += dr2;
            }
        }
    }
    {
        lock_guard<mutex> lock_msd(msd_mutex);
        for (int j = 0; j < n0; ++j) {
            msd[j] += local_msd[j];
        }
    }
}

void worker_thread(const array<array<double, cols>, rows>& dist_map,
                   const array<array<double, cols>, rows>& de_x_map,
                   const array<array<double, cols>, rows>& de_y_map,
                   int n0,
                   double dt,
                   vector<double>& msd,
                   double r_ref) {
    while (true) {
        int index;
        {
            unique_lock<mutex> lock(mtx);
            cv.wait(lock, [] { return !tasks.empty() || done; });

            if (done && tasks.empty())
                break;

            index = tasks.front();
            tasks.pop();
        }
        (void)index; // index 未使用，但保留任務語意
        simulate_trajectory(dist_map, de_x_map, de_y_map, n0, dt, msd, r_ref);
    }
}

int main() {
    vector<string> morphologies;
    int n0;
    double D0_ref;
    if (particle_size == 40){
        D0_ref = 0.088;
    }else if (particle_size == 100){
        D0_ref = 0.037;
    }else if (particle_size == 200){
        D0_ref = 0.02;
    }else{
        D0_ref = 0.02; // fallback
    }

    for (const auto& entry : fs::directory_iterator(folder+"/dist")) {
        if (entry.path().extension() == ".txt")
            morphologies.push_back(entry.path().string());
    }
    sort(morphologies.begin(), morphologies.end(), [](const string& a, const string& b) {
        return get_time(a) < get_time(b);
    });

    for (size_t idx = 0; idx < morphologies.size(); ++idx) {
        string filename = morphologies[idx];
        int t = get_time(filename);
        double exp_domain_size = t_to_size[t];
        double grid_size = exp_domain_size * L_inv; // um
        double r_ref = ((0.001 * R_ref) / grid_size); // convert nm to um and then to grid size

        double t_scale = grid_size * grid_size / D0_ref; // s
        double dt = compute_dt(t_scale, D0_ref);
        n0 = static_cast<int>(5.2 / (dt * t_scale)); // 5.2 seconds

        cout << "Processing: " << filename
             << " with exp domain size: " << exp_domain_size
             << " t_scale: " << t_scale
             << " dt: " << dt
             << " steps: " << n0 << endl;

        array<array<double, cols>, rows> dist_map = {};
        array<array<double, cols>, rows> de_x_map = {};
        array<array<double, cols>, rows> de_y_map = {};
        morphology_read(filename, dist_map, de_x_map, de_y_map, r_ref);

        string tag = fs::path(filename).stem().string();
        string out_folder = folder + "/"+ to_string(particle_size)+"nm_output";
        if (!fs::exists(out_folder)) {
            fs::create_directory(out_folder);
        }

        for (int b = 0; b < num_batches; ++b) {
            {
                unique_lock<mutex> lock(mtx);
                done = false;
                while (!tasks.empty()) tasks.pop();  // 清空上一輪任務
            }
            vector<double> msd (n0, 0.0);

            // Add tasks
            {
                unique_lock<mutex> lock(mtx);
                for (int i = 0; i < N; ++i) {
                    tasks.push(i);
                }
            }

            // Create worker threads
            vector<thread> threads;
            int thread_count = min(16, N);
            for (int i = 0; i < thread_count; ++i) {
                threads.emplace_back(worker_thread,
                                      cref(dist_map),
                                      cref(de_x_map),
                                      cref(de_y_map),
                                      n0,
                                      dt,
                                      ref(msd),
                                      r_ref);
            }

            {
                unique_lock<mutex> lock(mtx);
                done = true;
            }
            cv.notify_all();

            // Wait for threads to complete
            for (auto& th : threads) {
                th.join();
            }

            ofstream fout(out_folder+"/dist_chord_msd_" + tag + "_batch_" + to_string(b) + ".txt");
            for (int j = 0; j < n0; ++j) {
                double avg_msd = msd[j] * N_inv * tracer_inv;
                fout << j * dt << " " << avg_msd << "\n";
            }
            fout.close();
        }
    }
    return 0;
}
