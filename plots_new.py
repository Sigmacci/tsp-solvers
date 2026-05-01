
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_tspa_csv(path):
	# Reads TSPA.csv (semicolon-separated, x, y, reward)
	df = pd.read_csv(path, sep=';', header=None, names=['x', 'y', 'reward'])
	return df

def read_route_file(path):
	# Reads route file: one index per line (0-based)
	with open(path, 'r') as f:
		route = [int(line.strip()) for line in f if line.strip()]
	return route

def plot_tsp_solution(coords_df, route, show_rewards=True, save_path=None):
	coords = coords_df[['x', 'y']].values
	rewards = coords_df['reward'].values
	route = np.array(route)
	closed_route = np.append(route, route[0])
	ordered_coords = coords[closed_route]
	plt.figure(figsize=(12, 8))
	# Draw path
	plt.plot(ordered_coords[:, 0], ordered_coords[:, 1], '-o', color='tab:blue', alpha=0.7, label='Route')
	# Draw all cities
	plt.scatter(coords[:, 0], coords[:, 1], c='tab:red', s=40, edgecolors='white', zorder=3, label='Cities')
	# Highlight start/end
	plt.scatter([coords[route[0], 0]], [coords[route[0], 1]], c='tab:green', s=120, marker='*', label='Start/End', zorder=4)
	# Optionally show rewards as text
	# (Removed: no labels next to cities)
	plt.title('TSP Solution Visualization')
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.axis('equal')
	plt.grid(True, linestyle=':', alpha=0.5)
	if save_path:
		plt.savefig(save_path, bbox_inches='tight')
		print(f"Saved plot to {save_path}")
		plt.close()
	else:
		plt.show()

if __name__ == "__main__":
	# Paths (edit as needed)
	tspb_path = "input_data/TSPB.csv"
	route_path = "maybe_java/lab3/wyniki/route_najlepsza_ls_kandydackie.txt"
	output_path = "plots/tspb_route_najlepsza_ls_kandydackie.png"
	# Read data
	coords_df = read_tspa_csv(tspb_path)
	route = read_route_file(route_path)
	# Plot
	plot_tsp_solution(coords_df, route, show_rewards=True, save_path=output_path)
