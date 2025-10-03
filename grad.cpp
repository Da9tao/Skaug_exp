#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using Grid = std::vector<double>;

struct GridSize {
    std::size_t rows;
    std::size_t cols;
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

void compute_sobel(const Grid& input,
                   std::size_t rows,
                   std::size_t cols,
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

    const double scale = 1.0 / 8.0;

    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            double gx = 0.0;
            double gy = 0.0;

            for (int dr = -1; dr <= 1; ++dr) {
                const auto rr = static_cast<std::size_t>(std::clamp<int>(static_cast<int>(r) + dr, 0, static_cast<int>(rows) - 1));

                for (int dc = -1; dc <= 1; ++dc) {
                    const auto cc = static_cast<std::size_t>(std::clamp<int>(static_cast<int>(c) + dc, 0, static_cast<int>(cols) - 1));
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

void process_file(const fs::path& input_path, const fs::path& output_dir) {
    const Grid values = read_distance_field(input_path);
    const GridSize size = infer_dimensions(values);

    Grid grad_x(values.size(), 0.0);
    Grid grad_y(values.size(), 0.0);
    compute_sobel(values, size.rows, size.cols, grad_x, grad_y);

    const fs::path base_name = input_path.stem();
    write_grid(output_dir / (base_name.string() + "_grad_x.txt"), grad_x, size.rows, size.cols);
    write_grid(output_dir / (base_name.string() + "_grad_y.txt"), grad_y, size.rows, size.cols);
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

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_dist_file_or_dir> <output_dir>\n";
        return 1;
    }

    const fs::path input_path = argv[1];
    const fs::path output_dir = argv[2];

    try {
        if (!fs::exists(output_dir)) {
            fs::create_directories(output_dir);
        }

        const auto files = collect_inputs(input_path);
        for (const auto& file : files) {
            process_file(file, output_dir);
            std::cout << "Wrote gradients for " << file << " to " << output_dir << '\n';
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}

