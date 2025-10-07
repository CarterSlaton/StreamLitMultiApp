import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from scipy.signal import convolve2d
from PIL import Image
import io

# Page configuration
st.set_page_config(page_title="CNN Interactive Demo", layout="wide", page_icon="🧠")

# Title and introduction
st.title("🧠 Convolutional Neural Network (CNN) Interactive Demo")
st.markdown("""
This interactive application demonstrates how Convolutional Neural Networks process images step-by-step.
Explore each component of a CNN and see real-time visualizations!
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Demo:", [
    "🏠 Overview",
    "🎨 Image Dimensions",
    "🔍 Convolution Operation",
    "⚡ Activation Functions",
    "📉 Pooling Layers",
    "🎯 Complete CNN Pipeline"
])

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================
if page == "🏠 Overview":
    st.header("Understanding CNNs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("What are CNNs?")
        st.write("""
        Convolutional Neural Networks are specialized deep learning models designed for processing 
        grid-like data such as images. They automatically learn hierarchical features from raw pixels 
        to complex patterns.
        """)
        
        st.subheader("Key Components:")
        st.markdown("""
        - **Convolutional Layers**: Extract features using learnable filters
        - **Activation Functions**: Introduce non-linearity (ReLU)
        - **Pooling Layers**: Reduce spatial dimensions
        - **Fully Connected Layers**: Map features to predictions
        - **Output Layer**: Generate final classifications
        """)
    
    with col2:
        st.subheader("CNN Architecture Flow")
        
        # Create a simple architecture diagram
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Define layer positions and sizes
        layers = [
            ("Input\nImage", 5, 11, 2, 1.5, '#FFE5E5'),
            ("Conv +\nReLU", 5, 9, 2, 1, '#FFD4D4'),
            ("Pooling", 5, 7.5, 2, 0.8, '#FFC4C4'),
            ("Conv +\nReLU", 5, 6.2, 2, 1, '#FFB4B4'),
            ("Pooling", 5, 4.7, 2, 0.8, '#FFA4A4'),
            ("Flatten", 5, 3.5, 2, 0.6, '#FF9494'),
            ("Fully\nConnected", 5, 2.5, 2, 0.8, '#FF8484'),
            ("Output", 5, 1, 2, 0.8, '#FF7474'),
        ]
        
        for name, x, y, w, h, color in layers:
            rect = Rectangle((x-w/2, y-h/2), w, h, facecolor=color, 
                           edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, name, ha='center', va='center', fontsize=11, fontweight='bold')
            
        # Add arrows
        for i in range(len(layers)-1):
            y_start = layers[i][2] - layers[i][4]/2
            y_end = layers[i+1][2] + layers[i+1][4]/2
            ax.arrow(5, y_start, 0, y_end-y_start+0.1, 
                    head_width=0.3, head_length=0.15, fc='black', ec='black')
        
        st.pyplot(fig)
        plt.close()

# ============================================================================
# PAGE 2: IMAGE DIMENSIONS
# ============================================================================
elif page == "🎨 Image Dimensions":
    st.header("Understanding Image Dimensions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("RGB Color Images (3D)")
        st.write("A color image has three dimensions: Width × Height × Channels (3)")
        
        # Create a sample RGB visualization
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        
        # Create sample RGB image
        img_size = 50
        rgb_img = np.zeros((img_size, img_size, 3))
        rgb_img[:, :img_size//3, 0] = 1  # Red
        rgb_img[:, img_size//3:2*img_size//3, 1] = 1  # Green
        rgb_img[:, 2*img_size//3:, 2] = 1  # Blue
        
        # Show full RGB
        axes[0, 0].imshow(rgb_img)
        axes[0, 0].set_title('Full RGB Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Show R channel
        axes[0, 1].imshow(rgb_img[:,:,0], cmap='Reds')
        axes[0, 1].set_title('Red Channel', fontweight='bold', color='red')
        axes[0, 1].axis('off')
        
        # Show G channel
        axes[1, 0].imshow(rgb_img[:,:,1], cmap='Greens')
        axes[1, 0].set_title('Green Channel', fontweight='bold', color='green')
        axes[1, 0].axis('off')
        
        # Show B channel
        axes[1, 1].imshow(rgb_img[:,:,2], cmap='Blues')
        axes[1, 1].set_title('Blue Channel', fontweight='bold', color='blue')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.info(f"📊 Dimensions: {img_size} × {img_size} × 3 = {img_size*img_size*3:,} values")
    
    with col2:
        st.subheader("Grayscale Images (2D)")
        st.write("A grayscale image has two dimensions: Width × Height × Channels (1)")
        
        # Create grayscale visualization
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Create sample grayscale image with gradient
        gray_img = np.linspace(0, 1, img_size).reshape(1, -1).repeat(img_size, axis=0)
        
        ax.imshow(gray_img, cmap='gray')
        ax.set_title('Grayscale Image\n(Single Intensity Channel)', 
                    fontweight='bold', fontsize=14)
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(ax.imshow(gray_img, cmap='gray'), ax=ax, fraction=0.046)
        cbar.set_label('Intensity (0=Black, 1=White)', fontsize=10)
        
        st.pyplot(fig)
        plt.close()
        
        st.info(f"📊 Dimensions: {img_size} × {img_size} × 1 = {img_size*img_size:,} values")
    
    st.markdown("---")
    st.subheader("Key Differences")
    
    comparison = {
        "Aspect": ["Dimensions", "Channels", "Data Size", "Use Cases"],
        "RGB Color": ["Width × Height × 3", "3 (Red, Green, Blue)", 
                     "3× larger", "Photos, colored objects, natural scenes"],
        "Grayscale": ["Width × Height × 1", "1 (Intensity)", 
                     "Smaller", "Medical imaging, documents, some computer vision tasks"]
    }
    
    st.table(comparison)

# ============================================================================
# PAGE 3: CONVOLUTION OPERATION
# ============================================================================
elif page == "🔍 Convolution Operation":
    st.header("Convolution Operation: Feature Detection")
    
    st.write("""
    Convolution applies filters (kernels) to detect specific patterns in images. 
    Adjust the filter and see how it responds to different features!
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Select a Filter (Kernel)")
        
        filter_type = st.selectbox("Choose filter type:", [
            "Horizontal Edge Detection",
            "Vertical Edge Detection",
            "Blur",
            "Sharpen",
            "Custom"
        ])
        
        # Define filters
        filters = {
            "Horizontal Edge Detection": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
            "Vertical Edge Detection": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
            "Blur": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
            "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        }
        
        if filter_type == "Custom":
            st.write("Enter your custom 3×3 filter:")
            c1, c2, c3 = st.columns(3)
            kernel = np.zeros((3, 3))
            for i in range(3):
                for j, col in enumerate([c1, c2, c3]):
                    with col:
                        kernel[i, j] = st.number_input(f"[{i},{j}]", 
                                                      value=0.0, 
                                                      step=0.1, 
                                                      format="%.1f",
                                                      key=f"k_{i}_{j}")
        else:
            kernel = filters[filter_type]
        
        # Display kernel
        st.write("**Current Filter:**")
        fig, ax = plt.subplots(figsize=(3, 3))
        im = ax.imshow(kernel, cmap='RdBu', vmin=-2, vmax=2)
        ax.set_xticks([])
        ax.set_yticks([])
        
        for i in range(3):
            for j in range(3):
                text = ax.text(j, i, f'{kernel[i, j]:.1f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Input Pattern")
        
        pattern_type = st.selectbox("Choose input pattern:", [
            "Horizontal Lines",
            "Vertical Lines",
            "Diagonal Lines",
            "Checkerboard",
            "Circle"
        ])
        
        # Create input pattern
        size = 20
        if pattern_type == "Horizontal Lines":
            input_img = np.zeros((size, size))
            input_img[::3, :] = 1
        elif pattern_type == "Vertical Lines":
            input_img = np.zeros((size, size))
            input_img[:, ::3] = 1
        elif pattern_type == "Diagonal Lines":
            input_img = np.eye(size)
        elif pattern_type == "Checkerboard":
            input_img = np.indices((size, size)).sum(axis=0) % 2
        else:  # Circle
            y, x = np.ogrid[-size//2:size//2, -size//2:size//2]
            input_img = (x**2 + y**2 <= (size//3)**2).astype(float)
    
    # Perform convolution
    feature_map = convolve2d(input_img, kernel, mode='same', boundary='fill')
    
    # Display results
    st.markdown("---")
    st.subheader("Results")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input
    axes[0].imshow(input_img, cmap='gray')
    axes[0].set_title('Input Image', fontweight='bold', fontsize=14)
    axes[0].axis('off')
    
    # Kernel
    im1 = axes[1].imshow(kernel, cmap='RdBu', vmin=-2, vmax=2)
    axes[1].set_title('Filter (Kernel)', fontweight='bold', fontsize=14)
    axes[1].axis('off')
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f'{kernel[i, j]:.1f}',
                        ha="center", va="center", color="black", fontweight='bold')
    plt.colorbar(im1, ax=axes[1])
    
    # Output
    im2 = axes[2].imshow(feature_map, cmap='RdBu')
    axes[2].set_title('Feature Map (Output)', fontweight='bold', fontsize=14)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.info("""
    💡 **Observe:** Notice how the filter activates strongly where it matches the pattern! 
    - Edge detectors respond to edges
    - Blur filters smooth the image
    - Different filters detect different features
    """)

# ============================================================================
# PAGE 4: ACTIVATION FUNCTIONS
# ============================================================================
elif page == "⚡ Activation Functions":
    st.header("Activation Functions: Introducing Non-linearity")
    
    st.write("""
    Activation functions transform the output of neurons, introducing non-linearity 
    that allows neural networks to learn complex patterns.
    """)
    
    # Interactive plot
    x = np.linspace(-5, 5, 1000)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Select Activation Function")
        
        activation = st.radio("Choose function:", [
            "ReLU (Rectified Linear Unit)",
            "Sigmoid",
            "Tanh",
            "Leaky ReLU"
        ])
        
        st.markdown("---")
        
        if activation == "ReLU (Rectified Linear Unit)":
            st.markdown("""
            **ReLU: f(x) = max(0, x)**
            
            ✅ Advantages:
            - Simple and fast
            - Reduces vanishing gradient
            - Sparse activation
            
            ❌ Disadvantages:
            - Dead neurons (if x < 0)
            """)
            y = np.maximum(0, x)
            
        elif activation == "Sigmoid":
            st.markdown("""
            **Sigmoid: f(x) = 1 / (1 + e^(-x))**
            
            ✅ Advantages:
            - Outputs between 0 and 1
            - Smooth gradient
            
            ❌ Disadvantages:
            - Vanishing gradient problem
            - Not zero-centered
            """)
            y = 1 / (1 + np.exp(-x))
            
        elif activation == "Tanh":
            st.markdown("""
            **Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))**
            
            ✅ Advantages:
            - Zero-centered
            - Stronger gradients than sigmoid
            
            ❌ Disadvantages:
            - Still suffers from vanishing gradient
            """)
            y = np.tanh(x)
            
        else:  # Leaky ReLU
            alpha = st.slider("Alpha (slope for x < 0):", 0.01, 0.3, 0.1)
            st.markdown(f"""
            **Leaky ReLU: f(x) = max(αx, x)**
            
            (Current α = {alpha})
            
            ✅ Advantages:
            - Prevents dead neurons
            - Allows small gradient when x < 0
            """)
            y = np.where(x > 0, x, alpha * x)
    
    with col2:
        st.subheader("Function Visualization")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot function
        ax1.plot(x, y, linewidth=3, color='#FF6B6B')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Input (x)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Output f(x)', fontsize=12, fontweight='bold')
        ax1.set_title(f'{activation}', fontsize=14, fontweight='bold')
        
        # Plot gradient
        gradient = np.gradient(y, x)
        ax2.plot(x, gradient, linewidth=3, color='#4ECDC4')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Input (x)', fontsize=12, fontweight='bold')
        ax2.set_ylabel("Gradient f'(x)", fontsize=12, fontweight='bold')
        ax2.set_title('Gradient (Derivative)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Example with sample data
    st.markdown("---")
    st.subheader("See It In Action")
    
    st.write("Apply the activation function to sample convolution outputs:")
    
    sample_values = st.text_input("Enter comma-separated values:", "-2.5, -1.0, 0.0, 1.5, 3.0")
    
    try:
        values = np.array([float(v.strip()) for v in sample_values.split(',')])
        
        if activation == "ReLU (Rectified Linear Unit)":
            output = np.maximum(0, values)
        elif activation == "Sigmoid":
            output = 1 / (1 + np.exp(-values))
        elif activation == "Tanh":
            output = np.tanh(values)
        else:
            output = np.where(values > 0, values, alpha * values)
        
        result_df = {
            "Input": values,
            "Output": output,
            "Change": output - values
        }
        
        st.table(result_df)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 4))
        x_pos = np.arange(len(values))
        width = 0.35
        
        ax.bar(x_pos - width/2, values, width, label='Before Activation', alpha=0.8)
        ax.bar(x_pos + width/2, output, width, label='After Activation', alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Value Index', fontweight='bold')
        ax.set_ylabel('Value', fontweight='bold')
        ax.set_title('Before vs After Activation', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        st.pyplot(fig)
        plt.close()
        
    except:
        st.warning("Please enter valid numbers separated by commas")

# ============================================================================
# PAGE 5: POOLING LAYERS
# ============================================================================
elif page == "📉 Pooling Layers":
    st.header("Pooling Layers: Dimensionality Reduction")
    
    st.write("""
    Pooling reduces the spatial dimensions of feature maps while retaining important information.
    This makes the network more efficient and robust to small translations.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Pooling Configuration")
        
        pool_type = st.selectbox("Pooling Type:", ["Max Pooling", "Average Pooling"])
        pool_size = st.slider("Pool Size:", 2, 4, 2)
        
        st.markdown("---")
        
        if pool_type == "Max Pooling":
            st.info("""
            **Max Pooling** takes the maximum value from each region.
            
            ✅ Preserves the strongest activations
            ✅ More commonly used
            ✅ Provides translation invariance
            """)
        else:
            st.info("""
            **Average Pooling** takes the mean value from each region.
            
            ✅ Smoother downsampling
            ✅ Retains more spatial information
            ✅ Less prone to noise
            """)
    
    with col2:
        st.subheader("Input Feature Map")
        
        # Create sample feature map
        size = st.slider("Input Size:", 8, 16, 12)
        
        # Generate random feature map with some structure
        np.random.seed(42)
        feature_map = np.random.rand(size, size)
        
        # Add some structure (high activation regions)
        feature_map[2:5, 2:5] = np.random.rand(3, 3) + 1.5
        feature_map[7:10, 7:10] = np.random.rand(3, 3) + 1.2
    
    # Perform pooling
    pooled_size = size // pool_size
    pooled_map = np.zeros((pooled_size, pooled_size))
    
    for i in range(pooled_size):
        for j in range(pooled_size):
            region = feature_map[i*pool_size:(i+1)*pool_size, 
                               j*pool_size:(j+1)*pool_size]
            if pool_type == "Max Pooling":
                pooled_map[i, j] = np.max(region)
            else:
                pooled_map[i, j] = np.mean(region)
    
    # Display results
    st.markdown("---")
    st.subheader("Pooling Results")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Input
    im1 = axes[0].imshow(feature_map, cmap='viridis', interpolation='nearest')
    axes[0].set_title(f'Input Feature Map\n({size}×{size})', 
                     fontweight='bold', fontsize=14)
    axes[0].grid(True, which='both', color='white', linewidth=0.5)
    axes[0].set_xticks(np.arange(-0.5, size, pool_size), minor=False)
    axes[0].set_yticks(np.arange(-0.5, size, pool_size), minor=False)
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # Draw pooling regions
    for i in range(0, size, pool_size):
        for j in range(0, size, pool_size):
            rect = Rectangle((j-0.5, i-0.5), pool_size, pool_size,
                           linewidth=2, edgecolor='red', facecolor='none')
            axes[0].add_patch(rect)
    
    # Output
    im2 = axes[1].imshow(pooled_map, cmap='viridis', interpolation='nearest')
    axes[1].set_title(f'After {pool_type}\n({pooled_size}×{pooled_size})', 
                     fontweight='bold', fontsize=14)
    axes[1].grid(True, which='both', color='white', linewidth=0.5)
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Statistics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Input Size", f"{size}×{size}", f"{size*size} values")
    with col2:
        st.metric("Output Size", f"{pooled_size}×{pooled_size}", 
                 f"{pooled_size*pooled_size} values")
    with col3:
        reduction = (1 - (pooled_size*pooled_size)/(size*size)) * 100
        st.metric("Size Reduction", f"{reduction:.1f}%", 
                 f"↓ {size*size - pooled_size*pooled_size} values")

# ============================================================================
# PAGE 6: COMPLETE CNN PIPELINE
# ============================================================================
else:  # Complete CNN Pipeline
    st.header("🎯 Complete CNN Pipeline")
    
    st.write("""
    See how all components work together in a complete CNN! 
    Upload an image or use a sample to see the full processing pipeline.
    """)
    
    # Option to upload or use sample
    use_sample = st.checkbox("Use sample image", value=True)
    
    if use_sample:
        # Create a simple sample image
        img_array = np.zeros((28, 28, 3))
        # Draw a simple shape (circle)
        y, x = np.ogrid[-14:14, -14:14]
        mask = x**2 + y**2 <= 8**2
        img_array[mask] = [1, 0.5, 0.5]  # Reddish circle
        
        # Add some vertical lines
        img_array[:, 5:7, :] = [0.5, 0.5, 1]  # Bluish lines
        img_array[:, 21:23, :] = [0.5, 0.5, 1]
        
        input_image = img_array
    else:
        uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            image = image.resize((28, 28))
            input_image = np.array(image) / 255.0
            if len(input_image.shape) == 2:
                input_image = np.stack([input_image]*3, axis=-1)
        else:
            st.warning("Please upload an image or use the sample")
            st.stop()
    
    st.markdown("---")
    
    # Display the pipeline
    st.subheader("Processing Pipeline")
    
    # Stage 1: Input
    st.markdown("### Stage 1: Input Image")
    col1, col2 = st.columns([1, 2])
    with col1:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(input_image)
        ax.set_title('Original Image', fontweight='bold')
        ax.axis('off')
        st.pyplot(fig)
        plt.close()
    with col2:
        st.write(f"""
        **Input Shape:** {input_image.shape[0]} × {input_image.shape[1]} × {input_image.shape[2]}
        
        This is the raw input to our CNN. For RGB images, we have 3 channels (Red, Green, Blue).
        Each pixel has values between 0 and 1.
        """)
    
    st.markdown("---")
    
    # Stage 2: Convolution
    st.markdown("### Stage 2: Convolutional Layer + ReLU")
    
    # Apply multiple filters
    filters = {
        "Horizontal Edges": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
        "Vertical Edges": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        "Blur": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
    }
    
    feature_maps = []
    feature_maps_relu = []
    
    # Convert to grayscale for convolution
    gray_image = np.mean(input_image, axis=2)
    
    cols = st.columns(len(filters))
    for idx, (name, kernel) in enumerate(filters.items()):
        fm = convolve2d(gray_image, kernel, mode='same', boundary='fill')
        feature_maps.append(fm)
        
        # Apply ReLU
        fm_relu = np.maximum(0, fm)
        feature_maps_relu.append(fm_relu)
        
        with cols[idx]:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 6))
            
            im1 = ax1.imshow(fm, cmap='RdBu')
            ax1.set_title(f'{name}\n(Before ReLU)', fontsize=10, fontweight='bold')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046)
            
            im2 = ax2.imshow(fm_relu, cmap='viridis')
            ax2.set_title('After ReLU', fontsize=10, fontweight='bold')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    st.info(f"**Output Shape:** {fm_relu.shape[0]} × {fm_relu.shape[1]} × {len(filters)} feature maps")
    
    st.markdown("---")
    
    # Stage 3: Pooling
    st.markdown("### Stage 3: Max Pooling (2×2)")
    
    pooled_maps = []
    pool_size = 2
    
    cols = st.columns(len(feature_maps_relu))
    for idx, fm in enumerate(feature_maps_relu):
        h, w = fm.shape
        pooled_h, pooled_w = h // pool_size, w // pool_size
        pooled = np.zeros((pooled_h, pooled_w))
        
        for i in range(pooled_h):
            for j in range(pooled_w):
                region = fm[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
                pooled[i, j] = np.max(region)
        
        pooled_maps.append(pooled)
        
        with cols[idx]:
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(pooled, cmap='viridis')
            ax.set_title(f'{list(filters.keys())[idx]}\nPooled', 
                        fontsize=10, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
            st.pyplot(fig)
            plt.close()
    
    st.info(f"**Output Shape:** {pooled.shape[0]} × {pooled.shape[1]} × {len(pooled_maps)} (Reduced by {pool_size}×{pool_size})")
    
    st.markdown("---")
    
    # Stage 4: Flatten
    st.markdown("### Stage 4: Flatten")
    
    flattened = np.concatenate([pm.flatten() for pm in pooled_maps])
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write(f"""
        **Flattened Vector Length:** {len(flattened)}
        
        All pooled feature maps are concatenated into a single 1D vector.
        This prepares the data for fully connected layers.
        """)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 2))
        # Show first 100 values
        display_length = min(100, len(flattened))
        ax.bar(range(display_length), flattened[:display_length], width=1.0)
        ax.set_title(f'Flattened Feature Vector (showing first {display_length} of {len(flattened)} values)', 
                    fontweight='bold')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Stage 5: Fully Connected Layers
    st.markdown("### Stage 5: Fully Connected Layers")
    
    st.write("""
    In a real CNN, fully connected layers would process the flattened features.
    Here's a simplified simulation:
    """)
    
    # Simulate FC layers with random weights (for demonstration)
    np.random.seed(42)
    
    # FC Layer 1: flattened -> 64 neurons
    fc1_neurons = 64
    fc1_weights = np.random.randn(len(flattened), fc1_neurons) * 0.1
    fc1_output = np.maximum(0, np.dot(flattened, fc1_weights))  # ReLU activation
    
    # FC Layer 2: 64 -> 10 neurons (10 classes)
    num_classes = 10
    fc2_weights = np.random.randn(fc1_neurons, num_classes) * 0.1
    fc2_output = np.dot(fc1_output, fc2_weights)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(fc1_output)), sorted(fc1_output, reverse=True))
        ax.set_title('FC Layer 1 Activations (64 neurons)', fontweight='bold')
        ax.set_xlabel('Neuron Index (sorted by activation)')
        ax.set_ylabel('Activation Value')
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)
        plt.close()
        
        st.write(f"**Active Neurons:** {np.sum(fc1_output > 0)} / {fc1_neurons}")
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(num_classes), fc2_output)
        ax.set_title('FC Layer 2 Output (before softmax)', fontweight='bold')
        ax.set_xlabel('Class Index')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Stage 6: Output Layer
    st.markdown("### Stage 6: Output Layer (Softmax)")
    
    # Apply softmax
    exp_scores = np.exp(fc2_output - np.max(fc2_output))  # Numerical stability
    probabilities = exp_scores / np.sum(exp_scores)
    
    # Create class labels
    class_labels = [f"Class {i}" for i in range(num_classes)]
    predicted_class = np.argmax(probabilities)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(class_labels, probabilities, color=['#FF6B6B' if i == predicted_class else '#4ECDC4' 
                                                            for i in range(num_classes)])
        ax.set_xlabel('Probability', fontweight='bold', fontsize=12)
        ax.set_ylabel('Class', fontweight='bold', fontsize=12)
        ax.set_title('Final Class Probabilities', fontweight='bold', fontsize=14)
        ax.set_xlim(0, 1)
        
        # Add percentage labels
        for i, (label, prob) in enumerate(zip(class_labels, probabilities)):
            ax.text(prob + 0.02, i, f'{prob:.1%}', va='center', fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='x')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("### 🎯 Prediction")
        st.success(f"""
        **Predicted Class:** {class_labels[predicted_class]}
        
        **Confidence:** {probabilities[predicted_class]:.1%}
        """)
        
        st.write("**Top 3 Predictions:**")
        top3_indices = np.argsort(probabilities)[-3:][::-1]
        for idx in top3_indices:
            st.write(f"- {class_labels[idx]}: {probabilities[idx]:.1%}")
    
    st.markdown("---")
    
    # Summary
    st.markdown("## 📊 Pipeline Summary")
    
    summary_data = {
        "Stage": [
            "1. Input",
            "2. Convolution + ReLU",
            "3. Pooling",
            "4. Flatten",
            "5. Fully Connected 1",
            "6. Fully Connected 2",
            "7. Softmax Output"
        ],
        "Output Shape": [
            f"{input_image.shape[0]}×{input_image.shape[1]}×{input_image.shape[2]}",
            f"{fm_relu.shape[0]}×{fm_relu.shape[1]}×{len(filters)}",
            f"{pooled.shape[0]}×{pooled.shape[1]}×{len(pooled_maps)}",
            f"{len(flattened)} values",
            f"{fc1_neurons} neurons",
            f"{num_classes} neurons",
            f"{num_classes} probabilities"
        ],
        "Operation": [
            "Raw input image",
            "Apply filters → ReLU activation",
            "2×2 max pooling → reduce size",
            "Reshape to 1D vector",
            "Matrix multiplication → ReLU",
            "Matrix multiplication",
            "Convert to probabilities (sum=1)"
        ]
    }
    
    st.table(summary_data)
    
    st.success("""
    ### ✅ Key Takeaways:
    
    1. **Convolutional layers** detect local patterns using learnable filters
    2. **ReLU activation** introduces non-linearity for complex learning
    3. **Pooling** reduces dimensionality while keeping important features
    4. **Flattening** converts 2D feature maps to 1D for classification
    5. **Fully connected layers** combine features for decision making
    6. **Softmax** produces final probability distribution over classes
    
    This pipeline processes images from raw pixels to meaningful predictions!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>CNN Interactive Demo</strong> | Built with Streamlit</p>
    <p>Explore each component to understand how CNNs process images!</p>
</div>
""", unsafe_allow_html=True)