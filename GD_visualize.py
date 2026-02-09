import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# --- Page Config ---
st.set_page_config(page_title="GD Animator", layout="wide")

st.title("🎬 Real-Time Gradient Descent Animator")
st.markdown("Adjust the settings in the sidebar, then click **'▶️ Start Animation'** to watch the line learn!")

# --- Session State for Data ---
# We use session state so the random data doesn't change every time you click a button
if 'data_generated' not in st.session_state:
    st.session_state.X = np.array([])
    st.session_state.y = np.array([])
    st.session_state.true_m = 0
    st.session_state.true_b = 0
    st.session_state.data_generated = False

# --- Sidebar: Data Generation ---
st.sidebar.header("1. Data Settings")
n_points = st.sidebar.slider("Number of Data Points", 10, 100, 40)
noise_level = st.sidebar.slider("Noise Level", 0.0, 5.0, 1.5)

def generate_new_data():
    np.random.seed(int(time.time())) # Random seed based on time
    X = 2 * np.random.rand(n_points, 1)
    # Hidden true parameters (randomized slightly so it's not always the same)
    true_m = np.random.uniform(1.5, 4.0)
    true_b = np.random.uniform(1.0, 3.0)
    y = true_b + true_m * X + np.random.randn(n_points, 1) * noise_level
    
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.true_m = true_m
    st.session_state.true_b = true_b
    st.session_state.data_generated = True

if st.sidebar.button("🔄 Generate New Data"):
    generate_new_data()

# Generate initial data if none exists
if not st.session_state.data_generated:
    generate_new_data()

# --- Sidebar: Model & Animation ---
st.sidebar.header("2. Model & Animation")
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.3, 0.05, step=0.01)
epochs = st.sidebar.slider("Epochs (Iterations)", 10, 100, 50)
anim_delay = st.sidebar.slider("Animation Delay (sec)", 0.0, 1.0, 0.1)

st.sidebar.subheader("Initial Guess (Start Line)")
start_m = st.sidebar.slider("Start Slope (m)", -5.0, 10.0, 0.0)
start_b = st.sidebar.slider("Start Intercept (b)", -5.0, 10.0, 0.0)

st.sidebar.subheader("Constraints")
fix_slope = st.sidebar.checkbox("🔒 Fix Slope (m)")
fix_intercept = st.sidebar.checkbox("🔒 Fix Intercept (b)")

# --- Main Logic ---

# 1. Calculate Analytical Best Fit (The Target)
X = st.session_state.X
y = st.session_state.y
X_b = np.c_[np.ones((len(X), 1)), X]  # add x0 = 1
try:
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    best_m, best_b = theta_best[1][0], theta_best[0][0]
except:
    best_m, best_b = 0, 0 # Handle empty state

# Layout
col1, col2 = st.columns([3, 1])
plot_placeholder = col1.empty()
metrics_placeholder = col2.empty()

# Helper to plot
def plot_frame(history_m, history_b, current_loss, step):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Scatter Data
    ax.scatter(X, y, color='blue', alpha=0.4, label='Data')
    
    # X range for plotting lines
    x_range = np.array([[0], [2]])
    
    # 1. Best Fit (Green Dashed)
    y_best = best_m * x_range + best_b
    ax.plot(x_range, y_best, 'g--', linewidth=2, label='Best Fit (Target)')
    
    # 2. History (Faint Red Lines)
    # Only show every few steps to avoid clutter, but always show start
    for i, (hm, hb) in enumerate(zip(history_m[:-1], history_b[:-1])):
        if i == 0 or i % 5 == 0: 
            y_hist = hm * x_range + hb
            ax.plot(x_range, y_hist, color='red', alpha=0.15, linewidth=1)
            
    # 3. Current Gradient Descent Line (Solid Red)
    curr_m = history_m[-1]
    curr_b = history_b[-1]
    y_curr = curr_m * x_range + curr_b
    ax.plot(x_range, y_curr, 'r-', linewidth=3, label='Gradient Descent')

    ax.set_title(f"Step {step}/{epochs} | Loss: {current_loss:.4f}")
    ax.set_ylim(min(y.min(), 0) - 2, max(y.max(), 5) + 2)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    return fig

# Initial static view before animation
metrics_placeholder.metric("Current Loss", "Not Started")
fig = plot_frame([start_m], [start_b], 0, 0)
plot_placeholder.pyplot(fig)

# --- Animation Loop ---
start_btn = st.sidebar.button("▶️ Start Animation")

if start_btn:
    m = start_m
    b = start_b
    n = len(y)
    
    m_hist = [m]
    b_hist = [b]
    
    for epoch in range(epochs):
        # 1. Prediction
        y_pred = m * X + b
        
        # 2. Loss
        loss = (1/n) * np.sum((y_pred - y)**2)
        
        # 3. Gradients
        dm = (2/n) * np.sum(X * (y_pred - y))
        db = (2/n) * np.sum(y_pred - y)
        
        # 4. Update (unless fixed)
        if not fix_slope:
            m = m - learning_rate * dm
        if not fix_intercept:
            b = b - learning_rate * db
            
        m_hist.append(m)
        b_hist.append(b)
        
        # Update Plot
        fig = plot_frame(m_hist, b_hist, loss, epoch+1)
        plot_placeholder.pyplot(fig)
        plt.close(fig) # Prevent memory leak
        
        # Update Metrics
        with metrics_placeholder.container():
            st.subheader("Live Values")
            st.metric("Loss (MSE)", f"{loss:.4f}")
            st.markdown(f"**Slope (m):** `{m:.3f}`")
            st.markdown(f"**Intercept (b):** `{b:.3f}`")
            
            st.markdown("---")
            st.markdown("**Target (Best Fit):**")
            st.text(f"m: {best_m:.3f}")
            st.text(f"b: {best_b:.3f}")

        # Control Speed
        time.sleep(anim_delay)

    st.success("Optimization Complete!")
