# t-SNE

This project applies t-SNE (t-Distributed Stochastic Neighbor Embedding) to visualize the MNIST dataset. The implementation focuses on reducing the dimensionality of high-dimensional data into a 2D space for better interpretability.

# Mathematics

1. **Calculate Distance Matrix**  
    <img src="./figure/distance_equation.jpg">

2. **Find Sigma**  
    <img src="./figure/find_sigma_equation.jpg">

3. **Calculate P Matrix**  
    <img src="./figure/p_matrix_equation.jpg">

4. **Calculate Q Matrix**  
    <img src="./figure/q_matrix_equation.jpg">

5. **Calculate KL Divergence**  
    <img src="./figure/kl_divergence_equation.jpg">

6. **Calculate Gradient**  
    <img src="./figure/gradient_equation.jpg">

# Result

Results are visualized with two plots:  
1. KL Divergence over iterations to evaluate the optimization process. 
<img src="./figure/kl_divergence.jpg"> 
2. The final 2D t-SNE visualization of the MNIST dataset with color-coded labels.  
<img src="./figure/result.jpg"> 
