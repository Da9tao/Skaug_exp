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
const double Dx1 = 1;
const double Dy1 = 1;
const int rows = L, cols = L;
const int particle_size = 200; // nm
const double R_ref = particle_size * 0.5; //nm

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

void morphology_read(const string& filepath, array<array<int, cols>, rows>& space) {
    vector<float> data(rows * cols);
    ifstream ifs(filepath);
    size_t index = 0;
    float val;
    while (ifs >> val && index < data.size()) {
        data[index++] = val;
    }
    ifs.close();
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            space[j][i] = static_cast<int>(data[i * cols + j]);
}

void morphology_read_d(const string& dist_data, array<array<double, rows>, cols>& space) {
    vector<double> wwv(rows * cols);
    float wv;
    size_t index = 0;
    ifstream ifs(dist_data);
    while (ifs >> wv && index < rows * cols) {
        wwv[index++] = wv;
    }
    ifs.close();
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            space[i][j] = wwv[i * cols + j];
        }
    }
}

pair<int, int> get_new_coordinates(double x, double y) {
    int xd = static_cast<int>(round(x - L * round((x - L * 0.5) * L_inv))) % rows;
    int yd = static_cast<int>(round(y - L * round((y - L * 0.5) * L_inv))) % rows;
    return {xd, yd};
}

void simulate_trajectory(const array<array<double, cols>, rows>& space, int n0, double dt, vector<double>& msd, double& r_ref) {
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
        } while (space[xi][yi] == 0.0);

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

                    xs0 += k * dt * (-xs0) + sqrt(2.0 * Dx1 * dt) * u3;
                    ys0 += k * dt * (-ys0) + sqrt(2.0 * Dy1 * dt) * u4;

                    auto [xd, yd] = get_new_coordinates(xs0, ys0);
                    

                    if (space[xd][yd] <= r_ref) {
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
                        if (space[xd_new][yd_new] <= r_ref) {
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
            msd[j] += local_msd[j] ;
        }
    }
}

void worker_thread(const array<array<double, cols>, rows>& space, int n0, double dt, vector<double>& msd, double& r_ref) {
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
        simulate_trajectory(space, n0, dt, msd, r_ref);
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
        double grid_size = exp_domain_size * L_inv; //um
        double r_ref = ((0.001 * R_ref) / grid_size); // convert nm to um and then to grid size

        double dt = 0.0001;
        double t_scale = grid_size * grid_size / D0_ref; // s
        n0 = static_cast<int>(5.2 / (dt * t_scale)); // 5.2 seconds

        cout << "Processing: " << filename << " with exp domain size: " << exp_domain_size <<" t_scale: " << t_scale <<" steps: " << n0 << endl;

        array<array<double, cols>, rows> space = {};
        morphology_read_d(filename, space); 
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
            for (int i = 0; i < N; ++i) {
                tasks.push(i);
            }

            // Create worker threads
            vector<thread> threads;
            int thread_count = min(16, N);
            for (int i = 0; i < thread_count; ++i) {
                threads.emplace_back(worker_thread, ref(space), n0, dt, ref(msd), ref(r_ref));
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
