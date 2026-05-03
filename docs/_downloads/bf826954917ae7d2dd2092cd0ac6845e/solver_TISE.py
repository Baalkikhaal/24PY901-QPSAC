import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

# global variables
xmin, xmax = -15.0, +15.0
hbar=1
m_e=1
omega = 1.0

T = 1   # smoothening parameter for step function
V0 = 1  # step height
well_width = 10.0

# simulation parameters

h_0 = 1e-3
epsilon = 1e-3
psi_0, psi_1 = 0, epsilon

xs = np.arange(xmin, xmax, h_0)

# slider parameters
init_e = 1.0    # energy in a.u.
min_e = 0.0
max_e = 10.0
step_e = 1.0

fmin, fmax = -7, 2
init_f = 0.0
step_f = 1

# controls for potential

last_pot_index = 0

def smooth_step(x):
    return V0/(1 + np.exp(-1.0*x/T))

def inverse_smooth_step(x):
    return smooth_step(-1.0*x)
    
def finite_well(x):
    halfwidth = well_width/2.0
    return inverse_smooth_step(x+halfwidth) + smooth_step(x-halfwidth)

def finite_barrier(x):
    return -1.0*finite_well(x) + 1.0

def harmonic(x):
    halfwidth = well_width/2.0
    return 0.5*m_e*omega**2*(x/halfwidth)**2

# List of potentials
potentials = [lambda x: inverse_smooth_step(x), lambda x: finite_well(x), lambda x: finite_barrier(x), lambda x: harmonic(x)]
potential_names = ["step potential", "finite well", "finite barrier", "harmonic potential"]

def numerov(E, potential):
    """The initial two elements in the sequence {psi_n} can be chosen as 0 and epsilon"""
    psis = np.zeros(len(xs))
    # Get the ksquareds
    #ksquareds = np.array([2.0*m_e*(E - step(x))/hbar**2 for x in xs])
    ksquareds = np.array([2.0*m_e*(E - potential(x))/hbar**2 for x in xs])
    for i in range(len(psis)):
        if i == 0:
            psis[i] = psi_0
        elif i == 1:
            psis[i] = psi_1
        else:
            numer = 2.0*psis[i-1]*(1.0 - 5.0/12.0*h_0**2*ksquareds[i-1]) - psis[i-2]*(1.0 + 1.0/12.0*h_0**2*ksquareds[i-2])
            denom = 1 + 1.0/12.0*h_0**2*ksquareds[i]
            psis[i] = numer/denom
    return psis


# Get the psis

# Initial wave function
psi_0, psi_1 = 0, 1.0*epsilon

# Approximate wave function
E = 1.5
psis = np.zeros(len(xs))

#psis = numerov(E, inverse_smooth_step)
#pots = np.array([inverse_smooth_step(x) for x in xs])

potential = potentials[last_pot_index]
psis = numerov(E, potential)
pots = np.array([potential(x) for x in xs])

# Plot the wave
fig, ax = plt.subplots(figsize=(8, 6))
fig.canvas.manager.set_window_title("Numerical solver of Time Independent Schrodinger Equation")
plt.subplots_adjust(bottom=0.35)  # Space for slider

line1, = ax.plot(xs, psis, 'k-', label=r'$\psi(x)$')
ax2 = ax.twinx()
#line2, = ax2.plot(xs, pots, 'r--', label='inverse smooth step potential')
line2, = ax2.plot(xs, pots, 'r--', label='finite well')

ax.set_xlabel('x')
ax.set_ylabel(r'$\psi(x)$')
ax.legend()

ax2.set_ylabel('V(x)')
ax2.legend()

# Energy Slider axis and widget
ax_eslider = plt.axes([0.15, 0.2, 0.5, 0.03])
e_slider = Slider(ax_eslider, 'E (a.u.)', min_e, max_e, valinit=init_e, valstep=step_e, track_color='#0000FF80', initcolor='yellow')
e_slider.label.set_color('blue')

# multiplier widget
ax_multiplier = plt.axes([0.15, 0.1, 0.5, 0.03])
multiplier = Slider(ax_multiplier, 'f', fmin, fmax, valinit = init_f, valstep=step_f, track_color='#0000FF80',  initcolor='yellow')
multiplier.label.set_color('blue')

# Radio buttons for potentials
ax_pots_label = plt.axes([0.85, 0.25, 0.20, 0.06])
ax_pots_label.axis("off")
ax_pots_label.text(0, 0.5, "Potentials", va="center", fontsize=11)

ax_pots = plt.axes([0.75, 0.05, 0.20, 0.26])
rb_pots = RadioButtons(ax_pots, potential_names, active=last_pot_index)

# Update energy slider callback
def update_e(val):
    e = e_slider.val
    
    # Get the potential
    potential_label = rb_pots.value_selected
    index = 0
    for i in range(len(potentials)):
        if potential_names[i]==potential_label:
            index = i
    potential = potentials[index]
    # Get the updated psis
    psis = numerov(e, potential)
    
    # renormalize psis to [-1, 1]
    psis = psis/np.max(psis)
    line1.set_ydata(psis)
    
    #return line1
    fig.canvas.draw_idle()
    
def update_f(val):
    global e_slider
    f = multiplier.val
    old_init_e = e_slider.val
    old_emin = e_slider.valmin
    old_emax = e_slider.valmax
    old_step_e = e_slider.valstep
    
    # update the e slider
    e_slider.ax.clear()
    
    new_init_e = old_init_e
    new_emin = old_init_e 
    new_emax = new_init_e + 9*np.power(10.0, f)
    new_step_e = np.pow(10.0,f)
    
    # Create new slider with updated range
    e_slider = Slider(
        e_slider.ax,
        'E',
        new_emin,
        new_emax,
        valinit=new_init_e,
        valstep=new_step_e
    )
    
    e_slider.on_changed(update_e)

def update_pot(label):
    index = 0
    for i in range(len(potentials)):
        if potential_names[i]==label:
            index = i
            print(f'Index: {index}')
    potential = potentials[index]
    
    # Get the energy value
    E = e_slider.val
    # update the potential and wavefunction lines
    psis = numerov(E, potential)
    # renormalize psis to [-1, 1]
    psis = psis/np.max(psis)
    line1.set_ydata(psis)
    pots = np.array([potential(x) for x in xs])
    line2.set_ydata(pots)
    
    # update the label of line1
    line2.set_label(label)
    ax2.legend()

    fig.canvas.draw_idle()  # better to draw in idle mode when animations are not present.

# Callback function
e_slider.on_changed(update_e)
multiplier.on_changed(update_f)
rb_pots.on_clicked(update_pot)

plt.show()