import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
import math

# --- Helper Functions ---
def parse_input(x):
    """Parse user input, handling fractions and decimals"""
    try:
        return float(Fraction(str(x)))
    except:
        return float(x)

def to_fraction(x, max_den=1000):
    """Convert a number to fraction string if possible, otherwise keep as decimal"""
    try:
        frac = Fraction(x).limit_denominator(max_den)
        if abs(float(frac) - x) < 1e-10:
            return str(frac)
        else:
            return f"{x:.5f}"
    except:
        return f"{x:.5f}"

def display_radius(r):
    """Format radius display with fraction if possible"""
    try:
        frac = Fraction(r).limit_denominator(1000)
        if abs(float(frac) - r) < 1e-10:
            return f"r = {frac}"
        else:
            return f"r = {r:.5f}"
    except:
        return f"r = {r:.5f}"

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Circle Calculator App",
    page_icon="⭕",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Streamlit App ---
st.title("⭕ Circle Calculator App")
st.markdown("""
<p style='font-size:14px; color:gray;'>
&#128100; Abdullah Al Shakhee | Statistician & O-level Math Teacher | Data & Quantitative Analysis Enthusiast
</p>
""", unsafe_allow_html=True)

st.markdown("""
This app calculates circle equations and plots them based on different input methods.
Select a calculation type from the dropdown menu below.
""")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.info("""
    This app helps students and teachers visualize and calculate circle properties.
    Supports multiple input methods for circle definition.
    """)
    
    st.header("Instructions")
    st.markdown("""
    1. Select a calculation type
    2. Enter the required values
    3. Click the compute button
    4. View results and visualization
    """)
    
    st.header("Note")
    st.markdown("""
    - Fractions are supported (e.g., 1/2, 3/4)
    - Decimals are also accepted
    - Results show both exact fractions and decimal approximations
    """)

option = st.selectbox("Choose calculation type:",
                      ["1. General equation inputs",
                       "2. Circle from 3 points",
                       "3. Circle from 2 endpoints of diameter",
                       "4. Circle from center + 1 point",
                       "5. Circle with center on a line + 2 points"])

# Create figure with improved styling
fig, ax = plt.subplots(figsize=(7, 7))
ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.7)
ax.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.7)
ax.set_aspect("equal")
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_title('Circle Visualization')

# ----------------------------
if option == "1. General equation inputs":
    st.header("General Circle Equation")
    st.write("Enter coefficients for the general circle equation: x² + y² + Cx + Dy + E = 0")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        A = st.text_input("Coefficient of x² (A)", "1", help="Should equal coefficient of y²")
    with col2:
        B = st.text_input("Coefficient of y² (B)", "1", help="Should equal coefficient of x²")
    with col3:
        C = st.text_input("Coefficient of x (C)", "0")
    
    col4, col5 = st.columns(2)
    with col4:
        D = st.text_input("Coefficient of y (D)", "0")
    with col5:
        E = st.text_input("Constant (E)", "0")

    if st.button("Compute & Plot", key="type1"):
        try:
            A = parse_input(A); B = parse_input(B)
            C = parse_input(C); D = parse_input(D); E = parse_input(E)

            if abs(A - B) > 1e-10:
                st.error("⚠️ Not a circle (coefficients of x² and y² must be equal)")
            else:
                h = -C/(2*A)
                k = -D/(2*B)
                r_squared = h**2 + k**2 - E/A
                if r_squared < 0:
                    st.error("No real circle exists (negative radius squared)")
                else:
                    r = math.sqrt(r_squared)
                    
                    # Display results
                    st.success("Calculation successful!")
                    st.subheader("Results:")
                    st.write(f"**Equation:** {A}x² + {B}y² {C:+.3f}x {D:+.3f}y {E:+.3f} = 0")
                    st.write(f"**Center:** ({to_fraction(h)}, {to_fraction(k)})")
                    st.write(f"**Radius:** {display_radius(r)}")
                    
                    # Plot circle
                    theta = np.linspace(0, 2*np.pi, 600)
                    x = h + r*np.cos(theta)
                    y = k + r*np.sin(theta)
                    ax.plot(x, y, color='blue', linewidth=2, label="Circle")
                    ax.scatter(h, k, color="red", s=100, label="Center")
                    ax.legend()
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# ----------------------------
elif option == "2. Circle from 3 points":
    st.header("Circle Defined by Three Points")
    st.write("Enter three points on the circle (not collinear)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Point 1")
        X1 = st.text_input("x₁", "0")
        Y1 = st.text_input("y₁", "1")
    with col2:
        st.subheader("Point 2")
        X2 = st.text_input("x₂", "1")
        Y2 = st.text_input("y₂", "0")
    
    st.subheader("Point 3")
    X3 = st.text_input("x₃", "-1")
    Y3 = st.text_input("y₃", "0")

    if st.button("Compute & Plot", key="type2"):
        try:
            X1, Y1, X2, Y2, X3, Y3 = map(parse_input, [X1, Y1, X2, Y2, X3, Y3])
            A = np.array([[X1, Y1, 1], [X2, Y2, 1], [X3, Y3, 1]], dtype=float)
            B = -(np.array([X1**2+Y1**2, X2**2+Y2**2, X3**2+Y3**2], dtype=float))
            
            try:
                D, E, F = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                st.error("Points are collinear or not unique. Cannot form a circle.")
                st.stop()
                
            h = -D/2; k = -E/2; r = math.sqrt(h**2 + k**2 - F)
            
            # Display results
            st.success("Calculation successful!")
            st.subheader("Results:")
            st.write(f"**Equation:** x² + y² {D:+.3f}x {E:+.3f}y {F:+.3f} = 0")
            st.write(f"**Center:** ({to_fraction(h)}, {to_fraction(k)})")
            st.write(f"**Radius:** {display_radius(r)}")
            
            # Plot circle and points
            theta = np.linspace(0, 2*np.pi, 600)
            x = h + r*np.cos(theta); y = k + r*np.sin(theta)
            ax.plot(x, y, color='blue', linewidth=2, label="Circle")
            ax.scatter([X1, X2, X3], [Y1, Y2, Y3], color="red", s=100, label="Given Points")
            ax.scatter(h, k, color="green", s=100, label="Center")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# ----------------------------
elif option == "3. Circle from 2 endpoints of diameter":
    st.header("Circle with Given Diameter Endpoints")
    st.write("Enter the endpoints of the diameter")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Endpoint 1")
        X1 = st.text_input("x₁", "0", key="dia_x1")
        Y1 = st.text_input("y₁", "0", key="dia_y1")
    with col2:
        st.subheader("Endpoint 2")
        X2 = st.text_input("x₂", "2", key="dia_x2")
        Y2 = st.text_input("y₂", "0", key="dia_y2")

    if st.button("Compute & Plot", key="type3"):
        try:
            X1, Y1, X2, Y2 = map(parse_input, [X1, Y1, X2, Y2])
            h = (X1 + X2)/2; k = (Y1 + Y2)/2
            r = math.hypot(X2 - X1, Y2 - Y1)/2
            C = -2*h; D = -2*k; E = h**2 + k**2 - r**2
            
            # Display results
            st.success("Calculation successful!")
            st.subheader("Results:")
            st.write(f"**Equation:** x² + y² {C:+.3f}x {D:+.3f}y {E:+.3f} = 0")
            st.write(f"**Center:** ({to_fraction(h)}, {to_fraction(k)})")
            st.write(f"**Radius:** {display_radius(r)}")
            
            # Plot circle and points
            theta = np.linspace(0, 2*np.pi, 600)
            x = h + r*np.cos(theta); y = k + r*np.sin(theta)
            ax.plot(x, y, color='blue', linewidth=2, label="Circle")
            ax.scatter([X1, X2], [Y1, Y2], color="red", s=100, label="Endpoints")
            ax.scatter(h, k, color="green", s=100, label="Center")
            
            # Draw diameter line
            ax.plot([X1, X2], [Y1, Y2], color='orange', linestyle='--', label="Diameter")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# ----------------------------
elif option == "4. Circle from center + 1 point":
    st.header("Circle with Known Center and Point")
    st.write("Enter the center coordinates and a point on the circle")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Center")
        h = st.text_input("Center x", "0", key="center_x")
        k = st.text_input("Center y", "0", key="center_y")
    with col2:
        st.subheader("Point on Circle")
        X1 = st.text_input("Point x", "1", key="point_x")
        Y1 = st.text_input("Point y", "0", key="point_y")

    if st.button("Compute & Plot", key="type4"):
        try:
            h, k, X1, Y1 = map(parse_input, [h, k, X1, Y1])
            r = math.hypot(X1 - h, Y1 - k)
            C = -2*h; D = -2*k; E = h**2 + k**2 - r**2
            
            # Display results
            st.success("Calculation successful!")
            st.subheader("Results:")
            st.write(f"**Equation:** x² + y² {C:+.3f}x {D:+.3f}y {E:+.3f} = 0")
            st.write(f"**Center:** ({to_fraction(h)}, {to_fraction(k)})")
            st.write(f"**Radius:** {display_radius(r)}")
            
            # Plot circle and points
            theta = np.linspace(0, 2*np.pi, 600)
            x = h + r*np.cos(theta); y = k + r*np.sin(theta)
            ax.plot(x, y, color='blue', linewidth=2, label="Circle")
            ax.scatter([X1], [Y1], color="red", s=100, label="Given Point")
            ax.scatter(h, k, color="green", s=100, label="Center")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# ----------------------------
elif option == "5. Circle: center on line + 2 points":
    st.header("Circle with Center on a Line")
    st.write("Enter a line equation and two points on the circle")
    
    st.subheader("Line Equation")
    col1, col2 = st.columns(2)
    with col1:
        m = st.text_input("Slope (m)", "1")
    with col2:
        c_line = st.text_input("Intercept (c)", "0")
    
    st.subheader("Points on Circle")
    col3, col4 = st.columns(2)
    with col3:
        X1 = st.text_input("Point 1 x", "1", key="line_x1")
        Y1 = st.text_input("Point 1 y", "1", key="line_y1")
    with col4:
        X2 = st.text_input("Point 2 x", "2", key="line_x2")
        Y2 = st.text_input("Point 2 y", "0", key="line_y2")

    if st.button("Compute & Plot", key="type5"):
        try:
            m, c_line, X1, Y1, X2, Y2 = map(parse_input, [m, c_line, X1, Y1, X2, Y2])
            
            # Center (h,k) lies on line: y = m*h + c
            # Solve circle: (X1-h)^2 + (Y1-k)^2 = (X2-h)^2 + (Y2-k)^2
            # Solve for h
            denominator = 2 * (X1 - X2 + m * (Y2 - Y1))
            
            if abs(denominator) < 1e-10:
                st.error("Points and line do not form a unique circle or points are collinear.")
                st.stop()

            h = ((X1**2 + Y1**2 - X2**2 - Y2**2) + 2 * m * (Y2 - Y1)) / denominator
            k = m * h + c_line

            # Calculate radius
            r_squared = (X1 - h)**2 + (Y1 - k)**2
            if r_squared < 0:
                st.error("No real circle (imaginary radius).")
                st.stop()
            r = math.sqrt(r_squared)

            # Calculate coefficients for general equation
            C = -2 * h
            D = -2 * k
            E = h**2 + k**2 - r**2

            # Display results
            st.success("Calculation successful!")
            st.subheader("Results:")
            st.write(f"**Equation:** x² + y² {C:+.3f}x {D:+.3f}y {E:+.3f} = 0")
            st.write(f"**Center:** ({to_fraction(h)}, {to_fraction(k)})")
            st.write(f"**Radius:** {display_radius(r)}")

            # Plot circle, points, and line
            theta = np.linspace(0, 2*np.pi, 600)
            x = h + r*np.cos(theta); y = k + r*np.sin(theta)
            ax.plot(x, y, color='blue', linewidth=2, label="Circle")
            ax.scatter([X1, X2], [Y1, Y2], color="red", s=100, label="Given Points")
            ax.scatter(h, k, color="green", s=100, label="Center")
            
            # Plot the line
            line_x = np.array(ax.get_xlim())
            line_y = m * line_x + c_line
            ax.plot(line_x, line_y, color='orange', linestyle='--', 
                   label=f'Line: y = {to_fraction(m)}x + {to_fraction(c_line)}')
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<style>
.footer {
    text-align: center;
    color: gray;
    font-size: 14px;
}
</style>
<div class="footer">
    Circle Calculator App | Made with Streamlit
</div>
""", unsafe_allow_html=True)