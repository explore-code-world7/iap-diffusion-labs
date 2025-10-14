1. is u_{t}(x|z) gradient of p_{t}(x|z)?

2. in real scene, params to generate vector field remains unknown, how to generate such $\mu_{t}$, $\sigma_{t}$

3. why augment the reference marginal vector field $u_{t}^{ref}(x)$ with Langevin dynamics can add stochasticity while preserving the marginals?

4. flow-matching的作用是什么，从样本中估算conditonal vector field和conditional score,再根据ODE方程生成conditional path
* vector field是如何生成的?
* 对当前x和z变换，作为时刻ODE的梯度

5. why is flow-matching with linear probability paths popular?
* in their construction of the linear probability path, there is no need for $p_{\text{simple}}$ to be a Gaussian. Thus is more generalizable for generation.

6. 