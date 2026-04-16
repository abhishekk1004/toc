import itertools
import math
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

cities = {
    "Kathmandu": (27.7172, 85.3240),
    "Pokhara":   (28.2096, 83.9856),
    "Lumbini":   (27.4833, 83.2750),
    "Chitwan":   (27.5291, 84.3542),
    "Nagarkot":  (27.7167, 85.5167),
}

city_names = list(cities.keys())
n = len(city_names)

#  HAVERSINE DISTANCE — straight-line km between two GPS points

def haversine(city1, city2):
    R = 6371  # Earth radius in km
    lat1, lon1 = math.radians(city1[0]), math.radians(city1[1])
    lat2, lon2 = math.radians(city2[0]), math.radians(city2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# Build full distance matrix
dist = {(i, j): haversine(cities[i], cities[j]) for i in city_names for j in city_names}

#  ALGORITHM 1: BRUTE-FORCE (exact optimal, works for small n)


def brute_force_tsp(start="Kathmandu"):
    other = [c for c in city_names if c != start]
    best_route, best_dist = None, float("inf")
    all_routes = []

    for perm in itertools.permutations(other):
        route = [start] + list(perm) + [start]
        total = sum(dist[route[i], route[i + 1]] for i in range(len(route) - 1))
        all_routes.append((total, route))
        if total < best_dist:
            best_dist = total
            best_route = route

    all_routes.sort()   # sorted from shortest to longest
    return best_route, best_dist, all_routes


#  ALGORITHM 2: NEAREST-NEIGHBOUR HEURISTIC (fast approximation)


def nearest_neighbour_tsp(start="Kathmandu"):
    unvisited = set(city_names)
    route = [start]
    unvisited.remove(start)
    total = 0
    current = start

    while unvisited:
        nearest = min(unvisited, key=lambda c: dist[current, c])
        total += dist[current, nearest]
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    total += dist[current, start]   # return to start
    route.append(start)
    return route, total

# ─────────────────────────────────────────────────────────────────────────────
#  CONSOLE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def print_route(label, route, total):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    for i in range(len(route) - 1):
        d = dist[route[i], route[i + 1]]
        print(f"  {route[i]:<15} → {route[i+1]:<15}  ({d:6.1f} km)")
    print(f"  {'─'*45}")
    print(f"  Total distance: {total:.1f} km")

# ─────────────────────────────────────────────────────────────────────────────
#  RUN ALGORITHMS
# ─────────────────────────────────────────────────────────────────────────────

bf_route, bf_dist, all_routes = brute_force_tsp()
nn_route, nn_dist = nearest_neighbour_tsp()

print_route("BRUTE-FORCE (Optimal)", bf_route, bf_dist)
print_route("NEAREST-NEIGHBOUR (Heuristic)", nn_route, nn_dist)

print(f"\n  Difference: {nn_dist - bf_dist:.1f} km  "
      f"({(nn_dist - bf_dist) / bf_dist * 100:.1f}% worse than optimal)")

# ─────────────────────────────────────────────────────────────────────────────
#  VISUALIZATION — 6 panels
# ─────────────────────────────────────────────────────────────────────────────

# Longitude = x axis, Latitude = y axis
lons = {c: cities[c][1] for c in city_names}
lats = {c: cities[c][0] for c in city_names}

# Color palette
C = {
    'bg':      '#0d1117',
    'panel':   '#161b22',
    'grid':    '#21262d',
    'text':    '#e6edf3',
    'muted':   '#8b949e',
    'optimal': '#3fb950',   # green  — brute-force best
    'nn':      '#d29922',   # yellow — nearest-neighbour
    'city':    '#f78166',   # red-orange — regular cities
    'start':   '#bc8cff',   # purple — start/end city
    'accent':  '#58a6ff',   # blue   — algorithm complexity
}

def style_ax(ax):
    """Apply dark theme to an axis."""
    ax.set_facecolor(C['panel'])
    for spine in ax.spines.values():
        spine.set_edgecolor(C['grid'])
    ax.tick_params(colors=C['muted'], labelsize=8)

def draw_route_map(ax, route, route_color, title):
    """Draw a city map with an arrow route on top."""
    style_ax(ax)

    # Draw route arrows with segment distance labels
    for i in range(len(route) - 1):
        x0, y0 = lons[route[i]],   lats[route[i]]
        x1, y1 = lons[route[i+1]], lats[route[i+1]]
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=route_color,
                                   lw=2, mutation_scale=18))
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my, f"{dist[route[i], route[i+1]]:.0f} km",
                fontsize=7, color=C['muted'], ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', fc=C['bg'], ec='none', alpha=0.7))

    # Draw city dots
    for c in city_names:
        color = C['start'] if c == "Kathmandu" else C['city']
        ax.scatter(lons[c], lats[c], s=180, color=color, zorder=5,
                   edgecolors='white', linewidths=1)
        ax.text(lons[c] + 0.005, lats[c] + 0.012, c,
                fontsize=8.5, color=C['text'], fontweight='bold')

    ax.set_title(title, color=C['text'], fontsize=11, pad=10)
    ax.set_xlabel('Longitude', color=C['muted'], fontsize=8)
    ax.set_ylabel('Latitude',  color=C['muted'], fontsize=8)
    ax.grid(True, color=C['grid'], linewidth=0.5, alpha=0.5)

def cumulative_distances(route):
    """Build a list of cumulative distances along a route."""
    cum = [0]
    for i in range(len(route) - 1):
        cum.append(cum[-1] + dist[route[i], route[i + 1]])
    return cum


# ── Create figure ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14), facecolor=C['bg'])
fig.suptitle('Travelling Salesman Problem — Nepal Cities',
             fontsize=20, fontweight='bold', color='white', y=0.97)


# Optimal route map 
ax1 = fig.add_subplot(2, 3, 1)
draw_route_map(ax1, bf_route, C['optimal'],
               f'Optimal Route (Brute-Force)\nTotal: {bf_dist:.1f} km')


# Nearest-neighbour route map 
ax2 = fig.add_subplot(2, 3, 2)
draw_route_map(ax2, nn_route, C['nn'],
               f'Nearest-Neighbour Heuristic\nTotal: {nn_dist:.1f} km')


# bar + dot chart 
ax3 = fig.add_subplot(2, 3, 3)
style_ax(ax3)

route_dists = [r[0] for r in all_routes]
x_idx = np.arange(len(route_dists))

bar_colors = []
for i, d_val in enumerate(route_dists):
    if i == 0:
        bar_colors.append(C['optimal'])                    # best route
    elif abs(d_val - nn_dist) < 0.1:
        bar_colors.append(C['nn'])                         # nearest-neighbour
    else:
        bar_colors.append('#30363d')                       # all others

ax3.bar(x_idx, route_dists, color=bar_colors, width=0.7, zorder=3)
ax3.scatter(x_idx, route_dists, color='white', s=25, zorder=4, alpha=0.8)
ax3.axhline(bf_dist, color=C['optimal'], lw=1.5, linestyle='--', alpha=0.7)
ax3.axhline(nn_dist, color=C['nn'],      lw=1.5, linestyle='--', alpha=0.7)

ax3.set_title('All Possible Routes (24 permutations)\nSorted by Distance',
              color=C['text'], fontsize=11, pad=10)
ax3.set_xlabel('Route rank  (0 = shortest)', color=C['muted'], fontsize=8)
ax3.set_ylabel('Total Distance (km)',         color=C['muted'], fontsize=8)
ax3.grid(True, axis='y', color=C['grid'], linewidth=0.5, alpha=0.5)

legend_handles = [
    mpatches.Patch(color=C['optimal'], label=f'Optimal ({bf_dist:.0f} km)'),
    mpatches.Patch(color=C['nn'],      label=f'Nearest-Neighbour ({nn_dist:.0f} km)'),
    mpatches.Patch(color='#30363d',    label='Other routes'),
]
ax3.legend(handles=legend_handles, facecolor=C['bg'], edgecolor=C['grid'],
           labelcolor=C['text'], fontsize=8)



ax4 = fig.add_subplot(2, 3, 4)
style_ax(ax4)

matrix = np.array([[dist[c1, c2] for c2 in city_names] for c1 in city_names])
im = ax4.imshow(matrix, cmap='YlOrRd', aspect='auto')
short_names = [c[:5] for c in city_names]
ax4.set_xticks(range(n))
ax4.set_yticks(range(n))
ax4.set_xticklabels(short_names, rotation=30, ha='right', color=C['text'], fontsize=8)
ax4.set_yticklabels(short_names, color=C['text'], fontsize=8)

for i in range(n):
    for j in range(n):
        text_color = 'black' if matrix[i, j] > 150 else 'white'
        ax4.text(j, i, f"{matrix[i, j]:.0f}", ha='center', va='center',
                 fontsize=8, color=text_color, fontweight='bold')

plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04).ax.tick_params(colors=C['muted'])
ax4.set_title('Distance Matrix (km)', color=C['text'], fontsize=11, pad=10)


# ── Panel 5: Cumulative distance step-by-step ─────────────────────────────────
ax5 = fig.add_subplot(2, 3, 5)
style_ax(ax5)

bf_cum = cumulative_distances(bf_route)
nn_cum = cumulative_distances(nn_route)
steps  = list(range(len(bf_route)))
step_labels = [c[:4] for c in bf_route]

ax5.plot(steps, bf_cum, color=C['optimal'], marker='o', lw=2.5, markersize=8,
         label=f'Brute-Force ({bf_dist:.0f} km)', zorder=4)
ax5.plot(steps, nn_cum, color=C['nn'],      marker='s', lw=2.5, markersize=8,
         linestyle='--', label=f'Nearest-Neighbour ({nn_dist:.0f} km)', zorder=4)

for i in range(len(steps)):
    ax5.vlines(i, min(bf_cum[i], nn_cum[i]), max(bf_cum[i], nn_cum[i]),
               color=C['muted'], lw=0.8, alpha=0.4)

ax5.set_xticks(steps)
ax5.set_xticklabels(step_labels, rotation=20, ha='right', color=C['text'], fontsize=8)
ax5.set_xlabel('City visited',            color=C['muted'], fontsize=8)
ax5.set_ylabel('Cumulative distance (km)', color=C['muted'], fontsize=8)
ax5.set_title('Cumulative Distance Step-by-Step', color=C['text'], fontsize=11, pad=10)
ax5.legend(facecolor=C['bg'], edgecolor=C['grid'], labelcolor=C['text'], fontsize=8)
ax5.grid(True, color=C['grid'], linewidth=0.5, alpha=0.5)


# ── Panel 6: Algorithm complexity (why brute-force doesn't scale) ──────────────
ax6 = fig.add_subplot(2, 3, 6)
style_ax(ax6)

n_vals = np.arange(3, 13)
bf_ops = [math.factorial(v - 1) for v in n_vals]
nn_ops = [v ** 2               for v in n_vals]

ax6.semilogy(n_vals, bf_ops, color=C['city'],   marker='o', lw=2.5, markersize=9,
             label='Brute-Force: (n-1)!')
ax6.semilogy(n_vals, nn_ops, color=C['accent'], marker='s', lw=2.5, markersize=9,
             linestyle='--', label='Nearest-Neighbour: n²')

ax6.axvline(5, color=C['optimal'], lw=1.5, linestyle=':', alpha=0.8)
ax6.text(5.1, bf_ops[2] * 1.5, 'Our Nepal\nproblem (n=5)',
         color=C['optimal'], fontsize=7.5)

ax6.fill_between(n_vals, bf_ops, nn_ops, alpha=0.07, color=C['city'])
ax6.set_xticks(n_vals)
ax6.set_xlabel('Number of cities (n)',     color=C['muted'], fontsize=8)
ax6.set_ylabel('Operations (log scale)',   color=C['muted'], fontsize=8)
ax6.set_title("Algorithm Complexity\n(Why brute-force doesn't scale)",
              color=C['text'], fontsize=11, pad=10)
ax6.legend(facecolor=C['bg'], edgecolor=C['grid'], labelcolor=C['text'], fontsize=8)
ax6.grid(True, color=C['grid'], linewidth=0.5, alpha=0.4)


# ── Save & show ───────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0, 1, 0.95])
output_file = 'tsp_nepal_visualization.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\n  Chart saved → {output_file}")


plt.show()