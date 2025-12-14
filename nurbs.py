import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import re

INPUT_FOLDER_NAME = 'inputs'


def _natural_sort_key(text: str):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r'(\d+)', text)]


def get_input_dir() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, INPUT_FOLDER_NAME)


def list_input_files(input_dir: str, extensions=('.txt',)):
    if not os.path.isdir(input_dir):
        return []
    files = []
    for name in os.listdir(input_dir):
        full = os.path.join(input_dir, name)
        if os.path.isfile(full) and name.lower().endswith(tuple(ext.lower() for ext in extensions)):
            files.append(name)
    files.sort(key=_natural_sort_key)
    return files


def read_nurbs_header(filepath: str):
    """
    Đọc nhanh 4 dòng đầu (bỏ dòng trống/comment) để hiển thị preview trong menu.
    Trả về (p, q, rows, cols) hoặc None nếu lỗi/thiếu dữ liệu.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            header = []
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                header.append(line)
                if len(header) >= 4:
                    break
        if len(header) < 4:
            return None
        p = int(header[0])
        q = int(header[1])
        rows = int(header[2])
        cols = int(header[3])
        return p, q, rows, cols
    except Exception:
        return None


def choose_input_file(input_dir: str):
    """In menu console và cho người dùng chọn file input."""
    files = list_input_files(input_dir)
    if not files:
        print(f"Không tìm thấy file input nào trong thư mục: {input_dir}")
        print("Hãy chắc chắn bạn có thư mục 'inputs' cùng cấp với nurbs.py và trong đó có các file .txt")
        return None

    while True:
        print("\n=========================")
        print("   CHỌN FILE INPUT NURBS")
        print("=========================")
        for i, name in enumerate(files, start=1):
            meta = read_nurbs_header(os.path.join(input_dir, name))
            if meta is None:
                extra = "(không đọc được header)"
            else:
                p, q, rows, cols = meta
                extra = f"(p={p}, q={q}, grid={rows}x{cols})"
            print(f"[{i:2d}] {name:<20} {extra}")
        print("[0] Thoát")

        choice = input("Nhập số để chọn (vd: 1) hoặc gõ tên file: ").strip()
        if choice.lower() in ("0", "q", "quit", "exit"):
            return None

        # Chọn theo số
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(files):
                return files[idx - 1]
            print("Lựa chọn không hợp lệ. Hãy nhập số trong danh sách.")
            continue

        if choice in files:
            return choice

        matches = [f for f in files if choice.lower() in f.lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print("Tên bạn gõ khớp nhiều file:")
            for m in matches:
                print(f" - {m}")
            print("Hãy gõ rõ hơn hoặc chọn bằng số.")
            continue

        print("Không tìm thấy file phù hợp. Hãy thử lại.")


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
        knots[degree + 1: -degree] = np.linspace(0, 1, num_middle + 2)[1:-1]

    knots[-degree - 1:] = 1.0

    return knots


def find_span(n, p, u, knots):
    if u >= knots[n + 1]:
        return n
    if u <= knots[p]:
        return p
    low, high = p, n + 1
    mid = (low + high) // 2
    while u < knots[mid] or u >= knots[mid + 1]:
        if u < knots[mid]:
            high = mid
        else:
            low = mid
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
            denom = (right[r + 1] + left[j - r])
            temp = N[r] / denom if denom != 0 else 0
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
        with open(filepath, 'r', encoding='utf-8') as f:
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
                raise ValueError(
                    f"File thiếu dữ liệu. Cần {expected_points} điểm, nhưng chỉ tìm thấy {len(lines) - data_start_line}."
                )

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
    input_dir = get_input_dir()

    selected_name = choose_input_file(input_dir)
    if selected_name is None:
        print("Đã thoát.")
        return

    file_path = os.path.join(input_dir, selected_name)

    data = read_nurbs_file(file_path)
    if data is None:
        return

    p, q, control_points, weights = data
    n = control_points.shape[0] - 1
    m = control_points.shape[1] - 1

    print(f"-> Cấu hình: Bậc ({p}, {q}), Lưới điểm ({n + 1}x{m + 1})")

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

    ax.plot_surface(surface_x, surface_y, surface_z, cmap='jet', alpha=0.8, edgecolor='none')

    cp_x = control_points[:, :, 0].flatten()
    cp_y = control_points[:, :, 1].flatten()
    cp_z = control_points[:, :, 2].flatten()
    ax.scatter(cp_x, cp_y, cp_z, color='black', s=30, label='Control Points')

    for r in range(n + 1):
        ax.plot(
            control_points[r, :, 0],
            control_points[r, :, 1],
            control_points[r, :, 2],
            'k--',
            linewidth=0.5,
            alpha=0.5
        )
    for c in range(m + 1):
        ax.plot(
            control_points[:, c, 0],
            control_points[:, c, 1],
            control_points[:, c, 2],
            'k--',
            linewidth=0.5,
            alpha=0.5
        )

    ax.set_title(f"NURBS Render from {selected_name}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
