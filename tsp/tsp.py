import itertools
import math

# ─────────────────────────────────────────────
#  Nepal Places — Real-life TSP Example
#  Distances are approximate road distances (km)
# ─────────────────────────────────────────────

cities = {
    "Kathmandu": (27.7172, 85.3240),
    "Pokhara":   (28.2096, 83.9856),
    "Lumbini":   (27.4833, 83.2750),
    "Chitwan":   (27.5291, 84.3542),
    "Nagarkot":  (27.7167, 85.5167),
}

# ── Haversine formula: straight-line distance in km ──────────────────────────
def haversine(city1, city2):
    R = 6371  # Earth's radius in km
    lat1, lon1 = math.radians(city1[0]), math.radians(city1[1])
    lat2, lon2 = math.radians(city2[0]), math.radians(city2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ── Build a distance matrix ───────────────────────────────────────────────────
city_names = list(cities.keys())
n = len(city_names)

dist = {}
for i in city_names:
    for j in city_names:
        dist[i, j] = haversine(cities[i], cities[j])

# ── Brute-Force TSP (exact, works well for small n ≤ 10) ─────────────────────
def brute_force_tsp(start="Kathmandu"):
    other_cities = [c for c in city_names if c != start]
    best_route = None
    best_distance = float("inf")

    for perm in itertools.permutations(other_cities):
        route = [start] + list(perm) + [start]   # full cycle back to start
        total = sum(dist[route[i], route[i + 1]] for i in range(len(route) - 1))

        if total < best_distance:
            best_distance = total
            best_route = route

    return best_route, best_distance

# ── Nearest-Neighbour Heuristic (fast approximation for large n) ─────────────
def nearest_neighbour_tsp(start="Kathmandu"):
    unvisited = set(city_names)
    route = [start]
    unvisited.remove(start)
    total_dist = 0

    current = start
    while unvisited:
        nearest = min(unvisited, key=lambda c: dist[current, c])
        total_dist += dist[current, nearest]
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    # Return to start
    total_dist += dist[current, start]
    route.append(start)
    return route, total_dist

# ── Pretty printer ────────────────────────────────────────────────────────────
def print_route(label, route, total):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    for i in range(len(route) - 1):
        d = dist[route[i], route[i + 1]]
        print(f"  {route[i]:<15} → {route[i+1]:<15}  ({d:6.1f} km)")
    print(f"  {'─'*45}")
    print(f"  Total distance: {total:.1f} km")

# ── Run both algorithms ───────────────────────────────────────────────────────
bf_route, bf_dist = brute_force_tsp()
nn_route, nn_dist = nearest_neighbour_tsp()

print_route("BRUTE-FORCE (Optimal)", bf_route, bf_dist)
print_route("NEAREST-NEIGHBOUR (Heuristic)", nn_route, nn_dist)

# ── Distance matrix display ───────────────────────────────────────────────────
print(f"\n{'─'*55}")
print("  Distance Matrix (km, straight-line Haversine)")
print(f"{'─'*55}")
header = f"{'':>12}" + "".join(f"{c[:7]:>9}" for c in city_names)
print(header)
for c1 in city_names:
    row = f"  {c1:<10}" + "".join(f"{dist[c1,c2]:>9.1f}" for c2 in city_names)
    print(row)