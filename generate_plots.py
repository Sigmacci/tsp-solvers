import pandas as pd
import matplotlib.pyplot as plt

def plot_tsp_route(csv_path, tour_indices):
    df = pd.read_csv(csv_path, sep=';', header=None, usecols=[0, 1])
    
    coords = df.values 
    
    closed_tour = list(tour_indices) + [tour_indices[0]]

    ordered_coords = coords[closed_tour]
    x_path = ordered_coords[:, 0]
    y_path = ordered_coords[:, 1]

    plt.figure(figsize=(12, 8))
    
    plt.plot(x_path, y_path, color='#2c3e50', linewidth=1.5, alpha=0.8, zorder=1, label='Path')
    
    plt.scatter(coords[:, 0], coords[:, 1], color='#e74c3c', s=30, edgecolors='white', zorder=2)
    
    plt.scatter(x_path[0], y_path[0], color='#27ae60', s=120, marker='P', label='Start/End', zorder=3)

    plt.title("TSP Solution Visualization", fontsize=14)
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    
    plt.show()

a = [26, 103, 89, 163, 176]
plot_tsp_route('./input_data/TSPB.csv', a)