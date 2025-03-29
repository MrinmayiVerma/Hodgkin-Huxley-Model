import numpy as np
import matplotlib.pyplot as plt

dt = 0.01
T = 50
time = np.arange(0, T, dt)

Cm = 1.0
gNa = 120.0
gK = 36.0
gL = 0.3
ENa = 50.0
EK = -77.0
EL = -54.4

I_ext = np.zeros(len(time))
I_ext[500:1500] = 10

def alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
def beta_m(V): return 4.0 * np.exp(-0.0556 * (V + 65))
def alpha_h(V): return 0.07 * np.exp(-0.05 * (V + 65))
def beta_h(V): return 1 / (1 + np.exp(-(V + 35) / 10))
def alpha_n(V): return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
def beta_n(V): return 0.125 * np.exp(-0.0125 * (V + 65))

V = np.zeros(len(time))
V[0] = -65
m = alpha_m(V[0]) / (alpha_m(V[0]) + beta_m(V[0]))
h = alpha_h(V[0]) / (alpha_h(V[0]) + beta_h(V[0]))
n = alpha_n(V[0]) / (alpha_n(V[0]) + beta_n(V[0]))

m_vals, h_vals, n_vals = [m], [h], [n]

for t in range(1, len(time)):
    INa = gNa * (m**3) * h * (V[t-1] - ENa)
    IK = gK * (n**4) * (V[t-1] - EK)
    IL = gL * (V[t-1] - EL)
    
    dV = (I_ext[t] - (INa + IK + IL)) / Cm
    V[t] = V[t-1] + dt * dV
    
    m += dt * (alpha_m(V[t]) * (1 - m) - beta_m(V[t]) * m)
    h += dt * (alpha_h(V[t]) * (1 - h) - beta_h(V[t]) * h)
    n += dt * (alpha_n(V[t]) * (1 - n) - beta_n(V[t]) * n)

    m_vals.append(m)
    h_vals.append(h)
    n_vals.append(n)

plt.figure(figsize=(10, 6))
plt.plot(time, V, label="Membrane Potential (mV)")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.title("Hodgkin-Huxley Neuron Simulation")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time, m_vals, label="m (Na⁺ activation)")
plt.plot(time, h_vals, label="h (Na⁺ inactivation)")
plt.plot(time, n_vals, label="n (K⁺ activation)")
plt.xlabel("Time (ms)")
plt.ylabel("Gating Variable Value")
plt.title("Hodgkin-Huxley Gating Variables")
plt.legend()
plt.grid()
plt.show()
