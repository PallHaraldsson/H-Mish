# H-Mish

My hard_mish approximation times as fast as ReLU for all values, and is closer to original Mish than the approximation (continuous second derivative, including at -1.0; or -4.0 for hard_mish2), I fork from.

```julia
julia> function hard_mish(x) # x(x+1)^2 "between" ReLU
         l = x + one(x)
         if x >= zero(x)
           return x
         elseif x <= l
           return zero(x)
         else
           return l^2*x
         end
       end


julia> function hard_mish2(x)  # x(0.25*x+1)^2 "between" ReLU
         l = convert(typeof(x), 0.25)*x + one(x)
         if x >= zero(x)
           return x
         elseif x <= convert(typeof(x), -4)
           return zero(x)
         else
           return l^2*x
         end
       end



julia> @btime hard_mish(-0.5f0)
  0.024 ns (0 allocations: 0 bytes)
0.0f0
```

so I find likely to be better than:

Formula - *(x/2).min(2, max(0, x+2))*

<div style="text-align:center"><img src ="assets/hard_mish_graph.png"  width="450"/></div>
<p>
    <em>Figure 1. Hard Mish Activation Function</em>
</p>

### CIFAR-10: 

|Architecture|Swish|H-Mish|Mish|ReLU|
|:---:|:---:|:---:|:---:|:---:|
|ResNet-20|90.42%|92.57%|**92.68%**|91.8%|
