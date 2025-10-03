#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using Grid = std::vector<double>;

struct GridSize {
    std::size_t rows;
    std::size_t cols;
};

struct Config {
    double particle_size_nm = 200.0; // diameter in nm
    double D0_ref = 0.02;            // um^2 / s
    double sobel_scale = 1.0 / 8.0;
};

Grid read_distance_field(const fs::path& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open distance map: " + path.string());
    }

    Grid values;
    double value;
    while (input >> value) {
        values.push_back(value);
    }

    if (values.empty()) {
        throw std::runtime_error("Distance map is empty: " + path.string());
    }

    return values;
}

GridSize infer_dimensions(const Grid& values) {
    const std::size_t total = values.size();
    const std::size_t side = static_cast<std::size_t>(std::llround(std::sqrt(static_cast<double>(total))));

    if (side * side != total) {
        throw std::runtime_error("Distance map is not a square grid");
    }

    return {side, side};
}

std::vector<fs::path> collect_inputs(const fs::path& input) {
    std::vector<fs::path> files;

    if (fs::is_directory(input)) {
        for (const auto& entry : fs::directory_iterator(input)) {
            if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                files.push_back(entry.path());
            }
        }
        std::sort(files.begin(), files.end());
    } else if (fs::is_regular_file(input)) {
        files.push_back(input);
    } else {
        throw std::runtime_error("Input path is neither a file nor a directory: " + input.string());
    }

    if (files.empty()) {
        throw std::runtime_error("No distance maps found at: " + input.string());
    }

    return files;
}

int extract_time_stamp(const fs::path& file_path) {
    const std::string name = file_path.stem().string();
    const auto pos = name.find("t=");
    if (pos == std::string::npos) {
        throw std::runtime_error("Failed to parse time stamp from filename: " + file_path.string());
    }

    std::size_t start = pos + 2;
    std::size_t end = name.find_first_not_of("0123456789", start);
    const std::string time_str = name.substr(start, end - start);
    return std::stoi(time_str);
}

void compute_sobel(const Grid& input,
                   std::size_t rows,
                   std::size_t cols,
                   double scale,
                   Grid& grad_x,
                   Grid& grad_y) {
    static constexpr int kx[3][3] = {
        {1, 0, -1},
        {2, 0, -2},
        {1, 0, -1},
    };

    static constexpr int ky[3][3] = {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1},
    };

    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            double gx = 0.0;
            double gy = 0.0;

            for (int dr = -1; dr <= 1; ++dr) {
                const auto rr = static_cast<std::size_t>(
                    std::clamp<int>(static_cast<int>(r) + dr, 0, static_cast<int>(rows) - 1));

                for (int dc = -1; dc <= 1; ++dc) {
                    const auto cc = static_cast<std::size_t>(
                        std::clamp<int>(static_cast<int>(c) + dc, 0, static_cast<int>(cols) - 1));

                    const double value = input[rr * cols + cc];
                    gx += static_cast<double>(kx[dr + 1][dc + 1]) * value;
                    gy += static_cast<double>(ky[dr + 1][dc + 1]) * value;
                }
            }

            grad_x[r * cols + c] = gx * scale;
            grad_y[r * cols + c] = gy * scale;
        }
    }
}

void write_grid(const fs::path& path, const Grid& grid, std::size_t rows, std::size_t cols) {
    std::ofstream output(path);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open output file: " + path.string());
    }

    output.setf(std::ios::scientific);
    output.precision(10);

    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            output << grid[r * cols + c];
            if (c + 1 < cols) {
                output << ' ';
            }
        }
        if (r + 1 < rows) {
            output << '\n';
        }
    }
}

struct DiffusivityMaps {
    Grid parallel;
    Grid perpendicular;
    Grid effective_x;
    Grid effective_y;
};

DiffusivityMaps compute_diffusivities(const Grid& distances,
                                      const Grid& grad_x,
                                      const Grid& grad_y,
                                      double r_ref,
                                      double D0) {
    DiffusivityMaps result{
        Grid(distances.size(), 0.0),
        Grid(distances.size(), 0.0),
        Grid(distances.size(), 0.0),
        Grid(distances.size(), 0.0),
    };

    constexpr double eps = 1e-12;

    for (std::size_t idx = 0; idx < distances.size(); ++idx) {
        const double dist = distances[idx];
        if (dist <= eps || r_ref <= 0.0) {
            continue;
        }

        const double R_over_d = r_ref / dist;
        const double d_over_R = dist / r_ref;
        double Dp = 0.0;

        if (R_over_d <= 2.0) {
            const double Rz2 = R_over_d * R_over_d;
            const double Rz3 = Rz2 * R_over_d;
            const double Rz4 = Rz3 * R_over_d;
            const double Rz5 = Rz4 * R_over_d;
            Dp = D0 * (1.0 - 0.5625 * R_over_d + 0.125 * Rz3 - 45.0 * Rz4 / 256.0 - Rz5 / 6.0);
        } else {
            const double logz = std::log(d_over_R);
            const double denom = logz * logz - 4.325 * logz + 1.591;
            if (denom != 0.0) {
                Dp = -D0 * (2.0 * (logz - 0.9543)) / denom;
            }
        }

        const double numerator = 6.0 * d_over_R * d_over_R + 2.0 * d_over_R;
        const double denominator = 6.0 * d_over_R * d_over_R + 9.0 * d_over_R + 2.0;
        const double Dv = (denominator != 0.0) ? D0 * (numerator / denominator) : 0.0;

        result.parallel[idx] = Dp;
        result.perpendicular[idx] = Dv;

        const double gx = grad_x[idx];
        const double gy = grad_y[idx];
        const double grad_sq = gx * gx + gy * gy;

        if (grad_sq <= eps) {
            result.effective_x[idx] = Dp;
            result.effective_y[idx] = Dp;
            continue;
        }

        const double inv_grad = 1.0 / std::sqrt(grad_sq);
        const double nx = gx * inv_grad;
        const double ny = gy * inv_grad;

        const double nx_sq = nx * nx;
        const double ny_sq = ny * ny;

        // Projection of the axis onto the wall tangent.
        const double cos_x = 1.0 - nx_sq;
        const double cos_y = 1.0 - ny_sq;

        result.effective_x[idx] = Dp * cos_x + Dv * (1.0 - cos_x);
        result.effective_y[idx] = Dp * cos_y + Dv * (1.0 - cos_y);
    }

    return result;
}

void process_file(const fs::path& file,
                  const fs::path& output_dir,
                  const Config& cfg,
                  const std::map<int, double>& t_to_size) {
    const auto distances = read_distance_field(file);
    const auto size = infer_dimensions(distances);

    const int time_stamp = extract_time_stamp(file);
    const auto it = t_to_size.find(time_stamp);
    if (it == t_to_size.end()) {
        throw std::runtime_error("Missing domain size for time stamp: " + std::to_string(time_stamp));
    }

    const double domain_size_um = it->second;
    const double grid_size_um = domain_size_um / static_cast<double>(size.rows);

    const double particle_radius_nm = 0.5 * cfg.particle_size_nm;
    const double particle_radius_um = particle_radius_nm * 1e-3;
    const double r_ref = particle_radius_um / grid_size_um;

    Grid grad_x(distances.size(), 0.0);
    Grid grad_y(distances.size(), 0.0);
    compute_sobel(distances, size.rows, size.cols, cfg.sobel_scale, grad_x, grad_y);

    const auto maps = compute_diffusivities(distances, grad_x, grad_y, r_ref, cfg.D0_ref);

    const fs::path base = file.stem();
    write_grid(output_dir / (base.string() + "_Dp.txt"), maps.parallel, size.rows, size.cols);
    write_grid(output_dir / (base.string() + "_Dv.txt"), maps.perpendicular, size.rows, size.cols);
    write_grid(output_dir / (base.string() + "_De_x.txt"), maps.effective_x, size.rows, size.cols);
    write_grid(output_dir / (base.string() + "_De_y.txt"), maps.effective_y, size.rows, size.cols);

    std::cout << "Processed " << file << " with r_ref=" << r_ref << " grid units" << std::endl;
}

Config parse_config(int argc, char** argv) {
    Config cfg;

    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--particle-size" && i + 1 < argc) {
            cfg.particle_size_nm = std::stod(argv[++i]);
        } else if (arg == "--d0" && i + 1 < argc) {
            cfg.D0_ref = std::stod(argv[++i]);
        } else if (arg == "--sobel-scale" && i + 1 < argc) {
            cfg.sobel_scale = std::stod(argv[++i]);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
    }

    if (cfg.particle_size_nm <= 0.0) {
        throw std::runtime_error("Particle size must be positive");
    }
    if (cfg.D0_ref <= 0.0) {
        throw std::runtime_error("D0 must be positive");
    }

    return cfg;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_dist_file_or_dir> <output_dir> [--particle-size nm] [--d0 um^2_per_s] [--sobel-scale s]\n";
        return 1;
    }

    const fs::path input_path = argv[1];
    const fs::path output_dir = argv[2];

    try {
        const Config cfg = parse_config(argc, argv);

        if (!fs::exists(output_dir)) {
            fs::create_directories(output_dir);
        }

        const std::map<int, double> t_to_size = {
            {1000, 247.1080942}, {2000, 216.5761537}, {3000, 204.049624},
            {4000, 191.427262}, {5000, 184.5860421}, {6000, 174.8961444},
            {7000, 169.1660005}, {8000, 166.2407307}, {9000, 161.038961},
            {10000, 156.051519}, {30000, 120.5969061}, {50000, 105.975122},
            {70000, 98.08307894}, {100000, 87.42639647}, {200000, 72.80066049},
            {400000, 59.41720823}, {700000, 50.42732327},
        };

        const auto files = collect_inputs(input_path);
        for (const auto& file : files) {
            process_file(file, output_dir, cfg, t_to_size);
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}
