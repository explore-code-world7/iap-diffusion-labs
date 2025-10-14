1. is u_{t}(x|z) gradient of p_{t}(x|z)?

2. lec2预先定义分布，然后采样获取u_t(x|z), 在真实场景中，如何估计u_t(x|z)?
* flow matching也可以在真实图像，逐渐添加噪声来获取u_t(x|z);


3. why augment the reference marginal vector field $u_{t}^{ref}(x)$ with Langevin dynamics can add stochasticity while preserving the marginals?

4. flow-matching的作用是什么，从样本中估算conditonal vector field和conditional score,再根据ODE方程生成conditional path


5. why is flow-matching with linear probability paths popular?
* in their construction of the linear probability path, there is no need for $p_{\text{simple}}$ to be a Gaussian. Thus is more generalizable for generation.

6. what's classifier-free guidance

7. alpha、beta是自定义变量，只要满足alpha0=beta1=0, alpha_1=beta0=1即可

8. Simulator顺时间顺序step从图像到噪声，逆时间顺序step即从噪声到图像

9. classifier-free guidance=条件flow-matching

10. 还给了个没看过的UNet网络示例，不错~

11. flow-matching是从噪声或者任意分布(linear probability paths)生成目标分布;

12. conditional vector field和markov random filed的联系和区别?

13. vector field是如何生成的?
> 对当前x和z变换，作为时刻ODE的梯度
> 在生成过程中, p(xt|z)的alpha_t, beta_t可以指定, ut因此也可指定; 未知alpha_t, beta_t该如何获取ut?
* alpha_t就是指定的, p(x|z)假定为关于z的高斯分布N(\mu_t(z), \sigma_t(z)^2\ddotI)
* then ut(x|z)= f(\mu_{t}(z), \sigma_{t}(z), x)

14. conditional scores:
```python
    def conditional_score(self, x, z, t):
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        return (z * alpha_t - x) / beta_t ** 2
```
$$p(x|z)=N(x|\mu_{t}(x_{1}), \sigma_{t}(x_{1})^{2}I)=N(x|\alpha_{t}z, \beta_{t}^{2}I)$$
$$\nabla_x \log p_t(x|z) = \nabla_x N(x;\alpha_t z,\beta_t^2 I_d) = \frac{\alpha_t z - x}{\beta_t^2}.$$

15. flow-matching和score matching假设p(x|z)服从同一个分布，并检验该分布，理论上等价;
* 而且二者可以相互转化(通过推导), 得到ut(x)作为梯度即可迭代生成x;
$$u_t^{\text{ref}}(x) = a_tx + b_t\nabla \log p_t^{\text{ref}}(x).$$

16. classifier-free guidance: 这里的classifier=label, 对ut(x|y)可进行分解;

17. flow-matching的流程
- 任意定义psimple，从pdata来的样本x中，随机定义mu\sigma，可计算流分布\phi_t(x)，ut(x|x1);
- 可用于训练vt(x);
- 生成中psimple随机采样x，x1~目标分布pdata, 再利用vt(x)采样x；
- 用ut(x|z)迭代的，应该是标准形式;
- 在linear，可用x1的作为x0;

18. 样本是如何生成的?在真正生成的时候，x1=z~pdata不可见，如何生成x1?
* ut(x|z)中z~pdata, x~p(x|z)=\mu_t x+\sigma_t;
```python
class LearnedVectorFieldODE(ODE):
    def __init__(self, net: MLPVectorField):
        self.net = net

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(x, t)
```
* 从psimple中采样，利用ut(x|t)来生成样本
```python
ode = LearnedVectorFieldODE(flow_model)
simulator = EulerSimulator(ode)
# 从psimple随机采样
x0 = path.p_simple.sample(num_samples) # (num_samples, 2)
ts = torch.linspace(0.0, 1.0, num_timesteps).view(1,-1,1).expand(num_samples,-1,1).to(device) # (num_samples, nts, 1)
# 根据z计算ut(x|x1=z),步长迭代x, 这是利用目标vector field ut(x|z)计算出来的
xts = simulator.simulate_with_trajectory(x0, ts) # (bs, nts, dim)
```
* 为什么从psimple采样有效?
由于t=0时刻p0(x|x1)~psimple,与pdata无关，故可直接psimple获取x, 再用vt(x)迭代生成最终的x

19. flow-matching真的能训练吗?
* x1和x，如果预先不知道分布，该如何采样, x1直接从数据集取样本;
* x=x(t)=\phi(t), sample_conditional_path, 由样本z和psimple构造;
```python
def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
#         Samples from the conditional distribution p_t(x|z) = N(alpha_t * z, beta_t**2 * I_d)
    return self.alpha(t) * z + self.beta(t) * torch.randn_like(z)
```
$$u_t(x|z) = \left(\dot{\alpha}_t-\frac{\dot{\beta}_t}{\beta_t}\alpha_t\right)z+\frac{\dot{\beta}_t}{\beta_t}x.$$
* 采样之后按公式计算ut(x|z)，即可训练vt(x), 最后从psimple迭代生成pdata

## related papers
1. https://www.paperdigest.org/report/?id=advances-in-flow-matching-insights-from-icml-2025-papers 

# basic function
1. Simulator= use self.step() to generate trajectory til time t;
2. GaussianConditionalProbabilityPath:
* sample_conditional_variable = sample from pdata to x1=z;
* sample_conditional_path = generate \phi_t(x) (=blend of psimple and pdata)
* conditional_vector_field = generate vector filed for conditional path \phi_t(x) = ut(x|x1=z)
* conditional_score = log \nabla p(x|x1=z)

## class-pdata
```python
    @property
    def dim(self) -> int:
        return self.means.shape[1]

    @property
    def distribution(self):
        return D.MixtureSameFamily(
                mixture_distribution=D.Categorical(probs=self.weights, validate_args=False),
                component_distribution=D.MultivariateNormal(
                    loc=self.means,
                    covariance_matrix=self.covs,
                    validate_args=False,
                ),
                validate_args=False,
            )

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x).view(-1, 1)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample(torch.Size((num_samples,)))

    @classmethod
    def random_2D(
        cls, nmodes: int, std: float, scale: float = 10.0, seed = 0.0
    ) -> "GaussianMixture":
        torch.manual_seed(seed)
        means = (torch.rand(nmodes, 2) - 0.5) * scale
        covs = torch.diag_embed(torch.ones(nmodes, 2)) * std ** 2
        weights = torch.ones(nmodes)
        return cls(means, covs, weights)

    @classmethod
    def symmetric_2D(
        cls, nmodes: int, std: float, scale: float = 10.0,
    ) -> "GaussianMixture":
        angles = torch.linspace(0, 2 * np.pi, nmodes + 1)[:nmodes]
        means = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale
        covs = torch.diag_embed(torch.ones(nmodes, 2) * std ** 2)
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)
```

## class-conditionalprobabilitypath
```python
    # 
    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """
        Samples the conditioning variable z and label y
        Args:
            - num_samples: the number of samples
        Returns:
            - z: (num_samples, c, h, w)
            - y: (num_samples, label_dim)
        """
        return self.p_data.sample(num_samples)
    
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # contional path满足起止条件即可
        """
        Samples from the conditional distribution p_t(x|z)
        Args:
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - x: samples from p_t(x|z), (num_samples, c, h, w)
        """
        return self.alpha(t) * z + self.beta(t) * torch.randn_like(z)
        
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # vector_field maps variables to their gradients in ODE;
        """
        Evaluates the conditional vector field u_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, c, h, w)
        """ 
        alpha_t = self.alpha(t) # (num_samples, 1, 1, 1)
        beta_t = self.beta(t) # (num_samples, 1, 1, 1)
        dt_alpha_t = self.alpha.dt(t) # (num_samples, 1, 1, 1)
        dt_beta_t = self.beta.dt(t) # (num_samples, 1, 1, 1)

        return (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + dt_beta_t / beta_t * x

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_score: conditional score (num_samples, c, h, w)
        """ 
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        return (z * alpha_t - x) / beta_t ** 2
```

# simulator是ODE采样类
```python
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor):
        """
        Takes one simulation step
        Args:
            - xt: state at time t, shape (batch_size, dim)
            - t: time, shape ()
            - dt: time, shape ()
        Returns:
            - nxt: state at time t + dt
        """
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor):
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: initial state at time ts[0], shape (batch_size, dim)
            - ts: timesteps, shape (nts,)
        Returns:
            - x_fina: final state at time ts[-1], shape (batch_size, dim)
        """
        for t_idx in range(len(ts) - 1):
            t = ts[t_idx]
            h = ts[t_idx + 1] - ts[t_idx]
            x = self.step(x, t, h)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor):
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: initial state at time ts[0], shape (bs, dim)
            - ts: timesteps, shape (num_timesteps,)
        Returns:
            - xs: trajectory of xts over ts, shape (batch_size, num_timesteps, dim)
        """
        xs = [x.clone()]
        for t_idx in tqdm(range(len(ts) - 1)):
            t = ts[t_idx]
            h = ts[t_idx + 1] - ts[t_idx]
            x = self.step(x, t, h)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)
```

# restatement-1
* 讲述flow-matching
1. 首先，给定样本分布，如何生成轨迹数据pt(x)，和pt(x)梯度
* 不知道p(x)为什么学习困难, 
* 给定x1~psimple，训练pt(x|x1), pt(x|x_{1})鬼知道什么分布，假定为正态~N(mu_t(x_1), sigma_t(x_1))
* \phi_{t}(x)=\sigma_{t}(x_1)x+\mu_{t}(x_{1}), 容易计算，d \phi_{t}(x)/dt = ut(\phi_{t}(x)|x_1)=xxxx
* mut(x1), sigma_t(x1)如何表示? 只要满足[0,1], 和[x1, 0]的start-end condition即可, 可特殊化: mut(x1)=f(t)x1, sigma_t随便设, x~N by assumption
2. 之后，训练一个样本学习pt(x)到梯度的映射
3. 从随机样本出发，步长迭代生成phi_t(x)~q，按正态分布采样生成x

# diffusion-statement-1
1. 
