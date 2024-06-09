import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_theme()

cats_df = {}
particles_count = set()
for file in os.listdir():
    if not file.startswith("perf_") or not file.endswith(".csv"):
        continue

    df = pd.read_csv(file)

    # avoid numba compilation time
    df = df.iloc[1:]

    row = df.iloc[0]

    cats_df[(row["tree"], row["numba"], row["particles_count"])] = {
        "detect_contact": df["detect_contact"].mean(),
        "solve": df["solve"].mean(),
        "total": df["solve"].mean() + df["detect_contact"].mean(),
    }
    particles_count.add(row["particles_count"])

particles_count = sorted(list(particles_count))

plt.title("Contact detection time: KDTree vs Brute force", fontsize=16)
plt.ylabel("Contact detection time (s)", fontsize=14)
plt.xlabel("Number of particles", fontsize=14)
plt.plot(particles_count, [cats_df[(False, False, i)]["detect_contact"] for i in particles_count], label="Brute force")
plt.plot(particles_count, [cats_df[(True, False, i)]["detect_contact"] for i in particles_count], label="KDTree")
plt.plot(particles_count, [cats_df[(False, True, i)]["detect_contact"] for i in particles_count], label="Brute force + Numba")
plt.plot(particles_count, [cats_df[(True, True, i)]["detect_contact"] for i in particles_count], label="KDTree + Numba")
plt.legend()
plt.savefig("chart/contact_detection.pdf")
plt.clf()

plt.title("Contact detection time numba", fontsize=16)
plt.ylabel("Contact detection time (s)", fontsize=14)
plt.xlabel("Number of particles", fontsize=14)
plt.plot(particles_count, [cats_df[(False, True, i)]["detect_contact"] for i in particles_count], label="Numba")
plt.plot(particles_count, [cats_df[(True, True, i)]["detect_contact"] for i in particles_count], label="Numba + KDTree")
plt.legend()
plt.savefig("chart/contact_detection_numba.pdf")
plt.clf()

plt.title("Contact detection time no numba", fontsize=16)
plt.ylabel("Contact detection time (s)", fontsize=14)
plt.xlabel("Number of particles", fontsize=14)
plt.plot(particles_count, [cats_df[(False, False, i)]["detect_contact"] for i in particles_count], label="No numba")
plt.plot(particles_count, [cats_df[(True, False, i)]["detect_contact"] for i in particles_count], label="No numba + KDTree")
plt.legend()
plt.savefig("chart/contact_detection_no_numba.pdf")
plt.clf()

plt.title("Contact solving time: numba vs no numba", fontsize=16)
plt.ylabel("Contact solving time (s)", fontsize=14)
plt.xlabel("Number of particles", fontsize=14)
plt.plot(particles_count, [cats_df[(True, False, i)]["solve"] for i in particles_count], label="No numba")
plt.plot(particles_count, [cats_df[(True, True, i)]["solve"] for i in particles_count], label="Numba")
plt.legend()
plt.savefig("chart/contact_solving.pdf")
plt.clf()

plt.title("Total time numba vs no numba", fontsize=16)
plt.ylabel("Total time (s)", fontsize=14)
plt.xlabel("Number of particles", fontsize=14)
plt.plot(particles_count, [cats_df[(False, False, i)]["total"] for i in particles_count], label="No numba")
plt.plot(particles_count, [cats_df[(False, True, i)]["total"] for i in particles_count], label="Numba")
plt.plot(particles_count, [cats_df[(True, False, i)]["total"] for i in particles_count], label="No numba + KDTree")
plt.plot(particles_count, [cats_df[(True, True, i)]["total"] for i in particles_count], label="Numba + KDTree")
plt.legend()
plt.savefig("chart/total_time.pdf")
plt.clf()

plt.title("Total time numba", fontsize=16)
plt.ylabel("Total time (s)", fontsize=14)
plt.xlabel("Number of particles", fontsize=14)
plt.plot(particles_count, [cats_df[(False, True, i)]["total"] for i in particles_count], label="Numba")
plt.plot(particles_count, [cats_df[(True, True, i)]["total"] for i in particles_count], label="Numba + KDTree")
plt.legend()
plt.savefig("chart/total_time_numba.pdf")
plt.clf()

plt.title("Total time no numba", fontsize=16)
plt.ylabel("Total time (s)", fontsize=14)
plt.xlabel("Number of particles", fontsize=14)
plt.plot(particles_count, [cats_df[(False, False, i)]["total"] for i in particles_count], label="No numba")
plt.plot(particles_count, [cats_df[(True, False, i)]["total"] for i in particles_count], label="No numba + KDTree")
plt.legend()
plt.savefig("chart/total_time_no_numba.pdf")
plt.clf()
