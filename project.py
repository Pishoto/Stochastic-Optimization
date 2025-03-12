import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt

# --- Global Variables ---
weights = np.array([])
points = np.array([])
num_points = 0
banned_points = set()
available_points = set()
num_iterations = 0
valid_preds = np.array([])
dp_arr = np.array([])
prev = []
W = 0

# --- Start/End ---
def reset_globals():
    global weights, points, num_points, banned_points, available_points, num_iterations
    global valid_preds, dp_arr, prev, W

    weights = np.array([])
    points = np.array([])
    num_points = 0
    banned_points = set()
    available_points = set()
    num_iterations = 0
    valid_preds = np.array([])
    dp_arr = np.array([])
    prev = []
    W = 0

def initialize():
    global weights, available_points, banned_points, valid_preds, valid_succs, dp_arr, prev, W

    weights = np.ones(num_points, dtype=int)
    available_points = set(range(num_points))
    banned_points = set()
    valid_preds = [find_valid_preds(i) for i in range(len(points))]

    dp_arr = np.ones(len(points), dtype=int)  # Initialize dp_arr with 1s (copy starting weights)
    prev = [[] for _ in range(len(points))]  # Initialize prev with empty lists

    find_longest_chains()
    
def finish():
    global start_time

    print("\n--- Finished Sheet ---")
    print("W:", W)
    print("Number of Iterations:", num_iterations)
    print("---------------")

# --- I/O Functions ---
def plot_points():
    global points, weights, W

    plt.figure(figsize=(10, 6))
    plt.scatter(points[:, 0], points[:, 1], c=weights, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(label='Weight')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Points with Weights')
    plt.grid(True)
    plt.legend([f'W = {W}'])
    plt.show()

def generate_points_file(num_points_per_sheet, num_sheets=1, value_range=100):
    """
    Generate random points with integer values between 0 and value_range for multiple sheets.
    """
    with pd.ExcelWriter('points.xlsx', engine='openpyxl') as writer:
        for sheet_num in range(1, num_sheets + 1):
            x_values = np.random.randint(0, value_range + 1, num_points_per_sheet)
            y_values = np.random.randint(0, value_range + 1, num_points_per_sheet)

            # Create a DataFrame
            df = pd.DataFrame({
                'X': x_values,
                'Y': y_values
            })

            # Write to Excel file with a specific sheet name
            sheet_name = f'Sheet{sheet_num}'
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"Sheet '{sheet_name}' with {num_points_per_sheet} points created successfully.")

def convert(sorted_indices, original_df, processed_dfs, sheet_name):
    global weights

    sorted_weights = weights.copy()
        
    # Convert back to original order
    original_order_weights = np.zeros_like(sorted_weights)
    original_order_weights[sorted_indices] = sorted_weights
    
    # Assign weights back to original dataframe
    original_df['Weight'] = original_order_weights
    processed_dfs[sheet_name] = original_df

def write_output(all_dfs, output_file_path):
    with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
        for sheet_name, df in all_dfs.items():
            # Remove the original_index column before writing
            df = df.drop('original_index', axis=1)
            df.to_excel(writer, sheet_name=sheet_name, index=False)

def read_input(file_path):
    all_dfs = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
    processed_data = {}
    
    for sheet_name, df in all_dfs.items():
        points = df[['X', 'Y']].values
        df['original_index'] = np.arange(len(df))
        sorted_indices = np.lexsort((points[:, 1], points[:, 0]))
        sorted_points = points[sorted_indices]
        processed_data[sheet_name] = (df, sorted_points, sorted_indices)
    
    return processed_data

# --- Helper Functions ---
def find_valid_preds(i):
    """
    Finds valid predecessors for point i.
    """
    return np.where((points[:i, 0] < points[i, 0]) & (points[:i, 1] < points[i, 1]))[0]

def count_chains_containing_point(point_idx):
    """
    Counts how many potential maximum-weight chains could contain this point.
    Returns a score - lower means the point appears in fewer chains.
    """
    global points, weights
    
    x, y = points[point_idx]
    
    preds = np.where(
        (points[:point_idx, 0] < x) & 
        (points[:point_idx, 1] < y)
    )[0]
    
    succs = np.where(
        (points[point_idx + 1:, 0] > x) & 
        (points[point_idx + 1:, 1] > y)
    )[0] + point_idx + 1
    
    # Calculate score based on number of potential chains
    # Lower score means the point is in fewer potential chains
    score = len(preds) + len(succs)
    
    return score

def fix_chosen_point(chosen_point):
    global weights, dp_arr, available_points, banned_points

    weights[chosen_point] = 2
    dp_arr[chosen_point] = 2
    available_points.remove(chosen_point)
    banned_points.add(chosen_point)

# --- Important Functions ---
def find_longest_chains():
    global points, W, banned_points, available_points, valid_preds, weights, dp_arr, prev

    n = len(points)     
    
    for i in range(n):
        valid_preds_for_i = valid_preds[i]
        
        if len(valid_preds_for_i) > 0:
            # Calculate potential DP values for all valid predecessors
            potential_dp_values = dp_arr[valid_preds_for_i] + weights[i]
            
            # Find the maximum DP value and its corresponding indices
            max_dp_value = np.max(potential_dp_values)
            max_indices = np.where(potential_dp_values == max_dp_value)[0]
            
            # Update dp_arr and prev arrays
            dp_arr[i] = max_dp_value
            prev[i] = valid_preds_for_i[max_indices].tolist()   # Only preds that lead to max value

    longest_chain_length = max(dp_arr)
    W = longest_chain_length

    longest_chains_points = set()
    
    # Lambda function
    def add_to_banned_set(current):
        if current in longest_chains_points:
            return
        longest_chains_points.add(current)
        for p in prev[current]:
            add_to_banned_set(p)
    # ------------------------------

    indices_to_ban = np.where(dp_arr == W)[0]   # End points
    for i in indices_to_ban:
        add_to_banned_set(i)
        longest_chains_points.update(prev[i])

    banned_points.update(longest_chains_points)
    available_points.difference_update(banned_points)    

def fix_dp(point):
    global dp_arr, prev, available_points, banned_points

    n = len(points)

    for i in range(point, n):
        valid_preds_for_i = valid_preds[i]

        if len(valid_preds_for_i) > 0:
            potential_dp_values = dp_arr[valid_preds_for_i] + weights[i]
            max_dp_value = np.max(potential_dp_values)
            max_indices = np.where(potential_dp_values == max_dp_value)[0]
            dp_arr[i] = max_dp_value
            prev[i] = valid_preds_for_i[max_indices].tolist()
    
    longest_chains_points = set()
    
    def add_to_banned_set(current):
        if current in longest_chains_points:
            return
        
        longest_chains_points.add(current)

        for p in prev[current]:
            add_to_banned_set(p)

    for i in range(point, len(points)):
        if dp_arr[i] == W:
            add_to_banned_set(i)
            longest_chains_points.update(prev[i])

    banned_points.update(longest_chains_points)
    available_points.difference_update(banned_points)

# --- Algorithms ---
def random_algorithm():
    chosen_point = random.sample(available_points, 1)[0]    # [0] to get the integer
    return chosen_point

def greedy_algorithm():
    global available_points, weights, banned_points

    scores = [(point_idx, count_chains_containing_point(point_idx)) 
             for point_idx in available_points]
    
    chosen_point = max(scores, key=lambda x: x[1])[0]

    return chosen_point

def smallest_point_algorithm():
    min_point = min(available_points, key=lambda idx: (points[idx][0], points[idx][1]))
    return min_point

def middle_point_algorithm():
    middle_point = sorted(available_points, key=lambda idx: (points[idx][0], points[idx][1]))[len(available_points) // 2]
    return middle_point

def largest_point_algorithm():
    max_point = max(available_points, key=lambda idx: (points[idx][0], points[idx][1]))
    return max_point

def smart_choice_algorithm():
    """
    Select a point with highest dp_arr value that is not W.
    Expectation: Should ban more points each iteration.
    """
    max_point = max(available_points, key=lambda idx: dp_arr[idx] if dp_arr[idx] < W else -1)
    return max_point

# --- Main Function ---
def main(input_file, output_file):
    global start_time, num_points, num_iterations, points, valid_preds
    global dp_arr, prev, weights, available_points, banned_points, W

    start_time = time.time()
    # generate_points_file(100, 5, 1000)    # Generate and overwrite old file
    all_input_dfs = read_input(input_file)
    processed_dfs = {}
    
    for sheet_name, (original_df, sorted_points, sorted_indices) in all_input_dfs.items():
        sheet_time = time.time()
        print(f"Processing sheet: {sheet_name}")
        points = sorted_points
        num_points = len(points)        
        initialize()
        
        print("Running algorithm...")
        while available_points:
            # chosen_point = smallest_point_algorithm()
            # chosen_point = middle_point_algorithm()
            # chosen_point = largest_point_algorithm()
            # chosen_point = random_algorithm()
            # chosen_point = greedy_algorithm()
            chosen_point = smart_choice_algorithm()
            fix_chosen_point(chosen_point)
            fix_dp(chosen_point)
            num_iterations += 1

            # if num_iterations % 500 == 0:
            #     print("Number of Iterations:", num_iterations)
            #     time3 = time.time()
            #     print(f"Time for {num_iterations} iterations:", time3 - sheet_time)

        finish()
        print("Sheet Time: ", time.time() - sheet_time)
        convert(sorted_indices, original_df, processed_dfs, sheet_name)
        reset_globals()
        # plot_points()
    write_output(processed_dfs, output_file)

    exit(0)

if __name__ == "__main__":
    input_file = 'random_points.xlsx'
    output_file = 'random_points.xlsx'
    main(input_file, output_file)