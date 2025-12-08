import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

INPUT_FOLDER = 'inputs'
INPUT_FILENAME = 'input5.txt'

def generate_knot_vector(degree, num_control_points):
    """
    Tự động tạo Knot Vector dạng 'Clamped Uniform'.
    Đây là dạng phổ biến nhất trong thiết kế, giúp mặt đi qua các điểm mép.
    Công thức: [0]*degree -> [0..1] -> [1]*degree
    """
    n = num_control_points - 1
    m = n + degree + 1
    knots = np.zeros(m + 1)
    
    num_middle = m - 2 * degree
    if num_middle > 0:
        knots[degree + 1 : -degree] = np.linspace(0, 1, num_middle + 2)[1:-1]
        
    knots[-degree - 1 :] = 1.0
    
    return knots

def find_span(n, p, u, knots):
    if u >= knots[n + 1]: return n
    if u <= knots[p]: return p
    low, high = p, n + 1
    mid = (low + high) // 2
    while u < knots[mid] or u >= knots[mid + 1]:
        if u < knots[mid]: high = mid
        else: low = mid
        mid = (low + high) // 2
    return mid

def basis_funs(i, u, p, knots):
    N = np.zeros(p + 1)
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    N[0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - knots[i + 1 - j]
        right[j] = knots[i + j] - u
        saved = 0.0
        for r in range(j):
            temp = N[r] / (right[r + 1] + left[j - r]) if (right[r + 1] + left[j - r]) != 0 else 0
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved
    return N

def calculate_surface_point(n, p, U, m, q, V, P, W, u, v):
    uspan = find_span(n, p, u, U)
    Nu = basis_funs(uspan, u, p, U)
    vspan = find_span(m, q, v, V)
    Nv = basis_funs(vspan, v, q, V)
    
    num = np.zeros(3)
    den = 0.0
    
    for k in range(p + 1):
        for l in range(q + 1):
            u_idx = uspan - p + k
            v_idx = vspan - q + l
            
            weight = W[u_idx][v_idx]
            basis = Nu[k] * Nv[l] * weight
            
            num += basis * P[u_idx][v_idx]
            den += basis
            
    return num / den if den != 0 else np.zeros(3)


def read_nurbs_file(filepath):
    """
    Đọc file cấu hình và trả về các tham số cần thiết
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

    print(f"Đang đọc dữ liệu từ: {filepath} ...")
    
    try:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            p = int(lines[0])
            q = int(lines[1])
            rows = int(lines[2])
            cols = int(lines[3])
            
            control_points = np.zeros((rows, cols, 3))
            weights = np.zeros((rows, cols))

            data_start_line = 4
            expected_points = rows * cols
            
            if len(lines) - data_start_line < expected_points:
                raise ValueError(f"File thiếu dữ liệu. Cần {expected_points} điểm, nhưng chỉ tìm thấy {len(lines) - data_start_line}.")

            idx = 0
            for r in range(rows):
                for c in range(cols):
                    parts = lines[data_start_line + idx].split()
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    w = float(parts[3])
                    
                    control_points[r, c] = [x, y, z]
                    weights[r, c] = w
                    idx += 1
            
            return p, q, control_points, weights

    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return None


def main():
    file_path = os.path.join(INPUT_FOLDER, INPUT_FILENAME)
    
    data = read_nurbs_file(file_path)
    if data is None: return
    
    p, q, control_points, weights = data
    n = control_points.shape[0] - 1
    m = control_points.shape[1] - 1
    
    print(f"-> Cấu hình: Bậc ({p}, {q}), Lưới điểm ({n+1}x{m+1})")

    U = generate_knot_vector(p, n + 1)
    V = generate_knot_vector(q, m + 1)
    
    print("-> Đang tính toán bề mặt NURBS (Sampling)...")
    resolution = 40 
    u_vals = np.linspace(0, 1, resolution)
    v_vals = np.linspace(0, 1, resolution)
    
    surface_x = np.zeros((resolution, resolution))
    surface_y = np.zeros((resolution, resolution))
    surface_z = np.zeros((resolution, resolution))
    
    for i, u in enumerate(u_vals):
        for j, v in enumerate(v_vals):
            pt = calculate_surface_point(n, p, U, m, q, V, control_points, weights, u, v)
            surface_x[i, j] = pt[0]
            surface_y[i, j] = pt[1]
            surface_z[i, j] = pt[2]
            
    print("-> Đang hiển thị...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(surface_x, surface_y, surface_z, cmap='jet', alpha=0.8, edgecolor='none')
    
    cp_x = control_points[:, :, 0].flatten()
    cp_y = control_points[:, :, 1].flatten()
    cp_z = control_points[:, :, 2].flatten()
    ax.scatter(cp_x, cp_y, cp_z, color='black', s=30, label='Control Points')
    
    for r in range(n + 1):
        ax.plot(control_points[r, :, 0], control_points[r, :, 1], control_points[r, :, 2], 'k--', linewidth=0.5, alpha=0.5)
    for c in range(m + 1):
        ax.plot(control_points[:, c, 0], control_points[:, c, 1], control_points[:, c, 2], 'k--', linewidth=0.5, alpha=0.5)

    ax.set_title(f"NURBS Render from {INPUT_FILENAME}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    plt.show()

if __name__ == "__main__":
    main()